"""Rebuild paired results from complete runs, including output discrepancies."""
import csv
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent


def load(folder, name):
    data = json.loads((R / folder / (name + '.json')).read_text())
    if data['status'] != 'complete':
        raise ValueError((folder, name, data['status']))
    return data


def totals(data):
    rows = []
    for group in data['groups']:
        steps = [s for s in group['steps'] if s['batch'] == data['args']['batch']
                 and not s['cap'] and not s['clipped']]
        rows.append((sum(s['emitted'] for s in steps), sum(s['seconds'] for s in steps)))
    return np.array(rows, dtype=float)


def compare(folder, a='paper', b='ver01'):
    base, new = load(folder, a), load(folder, b)
    assert base['prompt_ids'] == new['prompt_ids']
    sa, sb = base['summary'], new['summary']
    aa, bb = totals(base), totals(new)
    rng = np.random.default_rng(9101)
    idx = rng.integers(0, len(aa), size=(4000, len(aa)))
    ax, bx = aa[idx].sum(1), bb[idx].sum(1)
    ratio = (bx[:, 0] / bx[:, 1]) / (ax[:, 0] / ax[:, 1]) - 1
    mismatch = [i for i, (x, y) in enumerate(zip(base['outputs'], new['outputs'])) if x != y]
    return dict(folder=folder, baseline=a, candidate=b, n=len(aa),
                baseline_summary=sa, candidate_summary=sb,
                tps_percent=100 * (sb['tps'] / sa['tps'] - 1),
                verify_percent=100 * (sb['verify_ms'] / sa['verify_ms'] - 1),
                prompt_bootstrap_tps_ci_percent=(100 * np.quantile(ratio, [.025, .975])).tolist(),
                same_outputs=not mismatch, mismatching_prompts=mismatch,
                same_warm_outputs=[g['outputs'] for g in base['warm_groups']] ==
                                  [g['outputs'] for g in new['warm_groups']],
                source_hashes_equal=base['source_hashes'] == new['source_hashes'])


def main():
    comparisons = [compare(folder) for folder in ('tree_r0', 'tree_r1', 'chain')]
    batch4 = compare('chain', 'b4_paper', 'b4_ver01')
    greedy, ar = load('greedy', 'tree'), load('greedy', 'ar')
    assert greedy['prompt_ids'] == ar['prompt_ids']
    mismatch = []
    for i, (a, b) in enumerate(zip(greedy['outputs'], ar['outputs'])):
        if a == b:
            continue
        pos = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        mismatch.append(dict(index=i, first_difference=pos,
                             tree_token=a[pos] if pos < len(a) else None,
                             ar_token=b[pos] if pos < len(b) else None,
                             common_prefix=greedy['prompt_ids'][i] + a[:pos]))
    campaigns = []
    for folder in ('tree_r0', 'tree_r1', 'chain', 'greedy'):
        rows = json.loads((R / folder / 'campaign.json').read_text())
        for row in rows:
            if row['status'] != 'complete' or row['exit_code'] != 0 or row['external_pids']:
                raise AssertionError((folder, row))
        campaigns.append(dict(folder=folder, complete=len(rows)))
    result = dict(scope='Saved 480 first-turn inputs; full-model Llama2-7B/AMD-135M on RTX4090. '
                        'Prompt bootstrap is conditional on each process pair; not independent process replicates.',
                  comparisons=comparisons, batch4_smoke=batch4,
                  greedy=dict(tree_summary=greedy['summary'], ar_summary=ar['summary'],
                              n=len(greedy['outputs']), mismatches=mismatch), campaigns=campaigns)
    (R / 'ANALYSIS.json').write_text(json.dumps(result, indent=2) + '\n')
    metric_rows, parameters = [], {}
    for folder, names in [('tree_r0', ['paper', 'ver01']), ('tree_r1', ['paper', 'ver01']),
                         ('chain', ['paper', 'ver01', 'b4_paper', 'b4_ver01']),
                         ('greedy', ['tree', 'ar'])]:
        for name in names:
            d = load(folder, name)
            s = d['summary']
            metric_rows.append(dict(run=folder+'/'+name, batch=d['args']['batch'],
                                   temperature=d['args']['temperature'], mode=d['args']['mode'],
                                   prompts=len(d['outputs']), AL=s['al'], cache_hit=s['hit'],
                                   TPS=s['tps'], verification_ms=s['verify_ms']))
            parameters[folder+'/'+name] = dict(kwargs=d['kwargs'], args=d['args'], env=d['env'])
    with (R/'PERFORMANCE.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(metric_rows)
    (R/'PARAMETERS.json').write_text(json.dumps(parameters, indent=2)+'\n')
    for row in comparisons + [batch4]:
        print(row['folder'], row['baseline'], row['candidate'], 'TPS%', row['tps_percent'],
              'Verify%', row['verify_percent'], 'CI', row['prompt_bootstrap_tps_ci_percent'],
              'mismatches', len(row['mismatching_prompts']))
    print('Greedy versus AR:', len(mismatch), 'of', len(greedy['outputs']))


if __name__ == '__main__':
    main()
