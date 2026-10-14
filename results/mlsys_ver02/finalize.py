"""Verify final source provenance and package release evidence without old rewrites."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = Path('/data/chokwans99/duet-ver02-release-final-20261014')
BASE = 'f12d0f1da6a2e4a26e9cb11ddcca54c189736d19'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    checks = json.loads((HERE / 'checks/CHECKS.json').read_text())
    prov = json.loads((HERE / 'checks/CHECKS_PROVENANCE.json').read_text())
    if not prov['source_unchanged'] or len(checks) != 24 or any(
            r['returncode'] or r['errors'] or r['failures'] or r['skipped'] or not r['tests']
            for r in checks):
        raise RuntimeError('Final regression not complete/clean')
    for path, expected in prov['source_hashes'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Source changed after final tests: ' + path)
    campaign = json.loads((RAW / 'campaign.json').read_text())
    plan = json.loads((HERE / 'SMOKE_PLAN.json').read_text())
    if len(campaign) != len(plan) or any(r['returncode'] or r['status'] != 'complete'
                                      or r['external_pids'] for r in campaign):
        raise RuntimeError('Functional campaign incomplete or interfered')
    sys.path.insert(0, str(ROOT / 'ssd/bench'))
    from mlsys_request_reporting import classify, question_summary
    smoke = []
    for job in plan:
        path = RAW / (job['name'] + '.json')
        j = json.loads(path.read_text())
        if j['status'] != 'complete':
            raise RuntimeError('Incomplete ' + job['name'])
        for p, expected in j['source_hashes'].items():
            if sha(ROOT / p) != expected or prov['source_hashes'][p] != expected:
                raise RuntimeError('Model/regression source mismatch: ' + p)
        for name, expected in j['adapter_hashes'].items():
            if sha(ROOT / 'ssd/bench' / name) != expected:
                raise RuntimeError('Benchmark source changed: ' + name)
        records = j['request_records']
        if records is None:
            if job['name'] != 'l2_ar_baseline' or j['request_population']['status'] != 'unavailable':
                raise RuntimeError('Unexpected unsupported request schema')
            guards = []
        else:
            for record in records:
                if classify(record) != record['classification']:
                    raise RuntimeError('Classification mismatch')
            if question_summary(records) != j['request_population']:
                raise RuntimeError('Population mismatch')
            guards = j['request_population']['context_guard_no_decode_uids']
        expected_guard = job['name'] == 'l2_context_guard'
        if (len(guards) != int(expected_guard)
                or len(j['outputs']) != (1 if expected_guard else 16)):
            raise RuntimeError('Unexpected functional population')
        if expected_guard and any(j['summary'][key] is not None
                                  for key in ('tps', 'al', 'hit', 'step_ms', 'verify_ms')):
            raise RuntimeError('Guard metrics must be undefined')
        smoke.append(dict(name=job['name'], status=j['status'],
            outputs=len(j['outputs']), output_tokens=sum(map(len, j['outputs'])),
            context_guard_no_decode_uids=guards,
            summary=j['summary'], request_population=j['request_population'],
            preparation=j['measurement']['preparation_all_decode'],
            source_matches_final_tests=True, raw_sha256=sha(path)))
    parity = json.loads((HERE / 'PAPER_PARITY.json').read_text())
    if parity['status'] != 'complete':
        raise RuntimeError('Missing paper parity')
    save(HERE / 'VALIDATION.json', dict(suites=len(checks), tests=sum(r['tests'] for r in checks),
        errors=0, failures=0, skipped=0, source_matches=True, model_runs=smoke,
        paper_parity=parity,
        scope='Functional release checks; no new performance improvement claim'))

    raw_files = sorted(p for p in RAW.rglob('*') if p.is_file())
    save(HERE / 'RAW_MANIFEST.json', {str(p.relative_to(RAW)):dict(bytes=p.stat().st_size,
        sha256=sha(p)) for p in raw_files})
    with tarfile.open(HERE / 'RAW_SMOKES.tar.gz', 'w:gz') as tar:
        for path in raw_files: tar.add(path, arcname=str(path.relative_to(RAW)))
    with tarfile.open(HERE / 'SOURCE.tar.gz', 'w:gz') as tar:
        for p in sorted(prov['source_hashes']): tar.add(ROOT / p, arcname=p)
    with tarfile.open(HERE / 'CHECK_LOGS.tar.gz', 'w:gz') as tar:
        for path in sorted((HERE / 'checks').glob('*')):
            if path.is_file(): tar.add(path, arcname='checks/' + path.name)
        tar.add(HERE / 'paper_parity.log', arcname='paper_parity.log')

    previous = json.loads((ROOT / 'results/mlsys_ver01/precommit_review_20261014/COMMIT_PLAN.json').read_text())
    changes = [p for p,h in previous['runtime_hashes'].items() if sha(ROOT/p) != h]
    if set(changes) != {'ssd/ssd/engine/model_runner.py', 'ssd/ssd/engine/verifier.py'}:
        raise RuntimeError('Unexpected inference change after prior review')
    artifacts = {str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(HERE.rglob('*')) if p.is_file()
        and p != HERE / 'MANIFEST.json' and '__pycache__' not in p.parts}
    save(HERE / 'MANIFEST.json', dict(base=BASE,
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        source_commit_at_packaging=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime_changes_since_prior_review=changes,
        tested_source_hashes=prov['source_hashes'],artifacts=artifacts,
        note='Code was tested by exact file hashes before committing. Old ver01 archives remain frozen.'))
    print('PASS: 24 suites,',sum(r['tests'] for r in checks),'tests,',len(plan),
          'model jobs, final source hashes match')


if __name__ == '__main__': main()
