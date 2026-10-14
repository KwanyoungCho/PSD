"""Fail closed unless all final merged-source validation evidence is present."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(name):
    return json.loads((HERE / name).read_text())


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def verify_sources(mapping):
    for name, expected in mapping.items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                f'Current source differs from tested source: {name}')


def main():
    rows = read('checks/CHECKS.json')
    require(len(rows) == 33, 'Expected all 33 suites')
    require(all(r['tests'] > 0 and r['returncode'] == 0 and not
                (r['failures'] or r['errors'] or r['skipped']) for r in rows), 'Regression failed/skipped')
    provenance = read('checks/CHECKS_PROVENANCE.json')
    require(provenance['source_unchanged'], 'Source changed during regression')
    verify_sources(provenance['source_hashes'])
    runner = ROOT / 'results/mlsys_ver01/extensions/check_suites.py'
    require(hashlib.sha256(runner.read_bytes()).hexdigest() == provenance['runner_sha256'], 'Runner changed')
    models = read('models/status.json')
    require(models['status'] == 'complete' and models['source_unchanged'], 'Native model checks incomplete')
    require(len(models['runs']) == 6 and all(r['returncode'] == 0 for r in models['runs']), 'Expected 6 successful native runs')
    verify_sources(models['source_hashes'])
    for run in models['runs']:
        raw = read('models/' + run['name'] + '.json')
        require(raw['status'] == 'complete', f'Native run incomplete: {run["name"]}')
        require(raw['source_hashes'] == models['source_hashes'], 'Native model source mismatch')
    adapter = read('adapter/status.json')
    require(adapter['status'] == 'complete' and adapter['source_unchanged'], 'Tree adapter checks incomplete')
    require(len(adapter['runs']) == 2 and all(r['returncode'] == 0 for r in adapter['runs']), 'Expected Last and Frontier')
    require(adapter['source_hashes'] == models['source_hashes'], 'Native/adapter source differs')
    for run in adapter['runs']:
        raw = read('adapter/' + run['selection'] + '.json')
        require(raw['status'] == 'complete' and raw['source_hashes'] == models['source_hashes'], 'Adapter output mismatch')
        for name, expected in raw['adapter_hashes'].items():
            require(hashlib.sha256((ROOT / 'ssd/bench' / name).read_bytes()).hexdigest() == expected, f'Adapter source changed: {name}')
    tree = read('TREE_RESULTS_CHECK.json')
    require(tree['status'] == 'pass' and tree['raw_files_checked'] == 14, 'Tree original result check incomplete')
    analysis = (HERE / 'analysis_tests.log').read_text()
    require('Ran 5 tests' in analysis and analysis.rstrip().endswith('OK'), 'Analysis regression incomplete')
    result = dict(state='complete_local_merge_validation', runtime_commit=models['commit'],
                  suites=len(rows), tests=sum(r['tests'] for r in rows), failures=0, errors=0, skips=0,
                  native_model_runs=6, greedy_parity=models['greedy_parity'],
                  adapter_runs=2, adapter_phase_forests={r['selection']:r['exported_phase_forests'] for r in adapter['runs']},
                  analysis_tests=5, original_tree_runs=14, original_tree_raw_hashes_checked=14,
                  original_tree_source_commit='091fa64', source_unchanged=True,
                  limitations=['RTX4090, Llama2-7B/AMD135M functional checks; not Blackwell70B certification.',
                               'Original full AL runs predate review normalization fixes; not rerun on merged runtime.',
                               'No new performance or B>1 AL claim from these short functional runs.'])
    (HERE / 'VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
