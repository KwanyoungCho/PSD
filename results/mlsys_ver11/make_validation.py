"""Summarize completed checks; never infer success from an incomplete run."""
import hashlib
import json
from pathlib import Path
from check_suites import SUITES, source_hashes

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    rows = read('checks/CHECKS.json')
    provenance = read('checks/CHECKS_PROVENANCE.json')
    assert [row['name'] for row in rows] == list(SUITES)
    assert all(row['tests'] > 0 and row['returncode'] == 0 and
               not any(row[key] for key in ('failures', 'errors', 'skipped'))
               for row in rows)
    assert provenance['source_unchanged'] and not provenance['source_status']
    assert provenance['source_hashes'] == source_hashes()
    cli = read('CLI_CHECKS.json')
    assert cli['passed'] and not cli['weights_loaded']
    forest = read('GENERATED_NODES.json')
    result = dict(
        status='complete', scope='parameter contract, regression checks, historical trace reanalysis',
        branch=provenance['git_branch'], tested_commit=provenance['git_commit'],
        suites=len(rows), tests=sum(row['tests'] for row in rows),
        failures=0, errors=0, skipped=0, source_unchanged=True,
        cuda_visible_devices=provenance['cuda_visible_devices'],
        cli_config_check=cli['passed'], cli_weights_loaded=cli['weights_loaded'],
        log_parser_compatibility=cli['old_and_new_log_parser'],
        historical_tree_runs=len(forest['runs']),
        historical_root_records=sum(run['records'] for run in forest['runs']),
        full_model_online_benchmark=False,
        performance_claim=False,
        policy_changes=[
            'Automatic local top-L equals wire N; topm normalization changes with L.',
            'G=M can change expansion versus old G>M experiments.',
            'DS continuation input cap replaces scale; omitted cap uses all roots.'
        ],
        sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                for name in ('checks/CHECKS.json', 'checks/CHECKS_PROVENANCE.json',
                             'CLI_CHECKS.json', 'CLI_HELP.txt', 'GENERATED_NODES.json')})
    (HERE/'VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
