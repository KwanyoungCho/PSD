"""Verify committed tree evidence and recompute question-level AL comparisons.

Needs NumPy, no CUDA/model/raw outputs. Optional --raw-root additionally checks
the 14 original run files. Never modifies the frozen experiment artifacts.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'results/mlsys_ver02/equal_budget_20261014'
FROZEN = '091fa64'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def paired(a, b, keys):
    keys = sorted(keys)
    x, y = np.array([a[q] for q in keys]), np.array([b[q] for q in keys])
    delta = y - x
    draws = delta[np.random.default_rng(143).integers(len(keys), size=(5000, len(keys)))].mean(1)
    return dict(questions=len(keys), before=float(x.mean()), after=float(y.mean()),
                delta=float(delta.mean()), relative_percent=float(100*(y.mean()/x.mean()-1)),
                ci95=np.quantile(draws, [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    artifacts = json.loads((DATA / 'FINAL_ARTIFACT_SHA256.json').read_text())
    for name, expected in artifacts.items():
        require(sha(DATA / name) == expected, f'Artifact SHA256 mismatch: {name}')
    sources = json.loads((DATA / 'SOURCE_SHA256.json').read_text())
    for name, expected in sources.items():
        blob = subprocess.check_output(['git', 'show', f'{FROZEN}:{name}'], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == expected, f'Frozen commit source mismatch: {name}')
    summary = json.loads((DATA / 'FINAL_ANALYSIS.json').read_text())
    results = json.loads((DATA / 'RESULTS.json').read_text())
    require(summary['state'] == 'complete' and summary['total_runs'] == 14, 'Incomplete final analysis')
    with (DATA / 'QUESTION_METRICS.csv').open() as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 6720, 'Expected 480 questions x 7 arms x 2 replicas')
    data = defaultdict(dict)
    for row in rows:
        key = (row['host'], row['arm'])
        q = row['question_id']
        require(q not in data[key], f'Duplicate question: {key}/{q}')
        data[key][q] = row
    require(len(data) == 14, 'Missing/extra host-arm group')
    with (DATA / 'FINAL_TABLE.csv').open() as f:
        table = list(csv.DictReader(f))
    names = ('al', 'hit_percent', 'miss_al', 'p1_al', 'p2_al', 'miss_share', 'p1_share', 'p2_share')
    for row in table:
        host, arm = row['server'], row['method']
        values = data[(host, arm)]
        require(len(values) == 480, f'Question count mismatch: {host}/{arm}')
        for metric in names:
            a = np.array([float(q[metric]) for q in values.values()])
            a = a[np.isfinite(a)]
            original = results[host]['arms'][arm]['question_quality']['all'][metric]
            require(len(a) == original['questions'], f'Conditional denominator mismatch: {metric}')
            require(np.isclose(a.mean(), float(row[metric]), atol=1e-12, rtol=0), f'Table mismatch: {host}/{arm}/{metric}')
            require(np.isclose(a.mean(), original['mean'], atol=1e-12, rtol=0), f'Results mismatch: {metric}')
    keys = set(next(iter(data.values())))
    require(all(set(v) == keys for v in data.values()), 'Different question sets')
    excluded = {q for v in data.values() for q, r in v.items() if int(r['context_guard'])}
    require(excluded == set(summary['guard_sensitivity']['excluded_questions']), 'Guard population mismatch')
    pooled = {arm: {q: np.mean([float(data[(host, arm)][q]['al']) for host in ('local', 'remote')]) for q in keys}
              for host, arm in data if host == 'local'}
    computed = {}
    for name, expected in summary['comparisons'].items():
        a, b = name.split(' -> ')
        for label, population in (('paired_question_average_over_replicas', keys),
                                  ('without_any_context_guard_question', keys - excluded)):
            actual = paired(pooled[a], pooled[b], population)
            for metric, value in actual.items():
                require(np.allclose(value, expected[label][metric], atol=1e-12, rtol=0), f'Paired statistic mismatch: {name}/{label}/{metric}')
            if label == 'paired_question_average_over_replicas':
                computed[name] = actual
    checked_raw = 0
    if args.raw_root:
        old_root = Path('/data/chokwans99/equal-tree-paper-full-20261014')
        for run in summary['raw_manifest']:
            path = args.raw_root / Path(run['path']).relative_to(old_root)
            require(sha(path) == run['sha256'], f'Raw SHA256 mismatch: {path}')
            checked_raw += 1
    report = dict(status='pass', frozen_commit=FROZEN, source_files=len(sources),
                  hashed_artifacts=len(artifacts), question_rows=len(rows), runs=len(data),
                  questions=480, turns=summary['total_turns'], guard_free_questions=len(keys-excluded),
                  paired_comparisons=len(computed), raw_files_checked=checked_raw,
                  comparisons=computed,
                  scope='Question-level recomputation; per-step recount needs optional raw files and original analysis scripts.')
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}, indent=2))


if __name__ == '__main__':
    main()
