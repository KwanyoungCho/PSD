"""Prepare reviewable layers without staging, committing or rewriting history."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = Path('/data/chokwans99/PSD-tree-frozen-20261014')
EXPERIMENTAL = 'ssd/ssd/engine/helpers/packed_tree_forward.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout_runtime(root):
    return {str(p.relative_to(root)): p.read_text()
            for p in (root / 'ssd/ssd').rglob('*.py')}


def patch(a, b, name):
    changes = []
    lines = []
    for path in sorted(a.keys() | b.keys()):
        if a.get(path) == b.get(path):
            continue
        changes.append(path)
        lines.extend(difflib.unified_diff(a.get(path, '').splitlines(True),
                     b.get(path, '').splitlines(True),
                     fromfile='a/' + path if path in a else '/dev/null',
                     tofile='b/' + path if path in b else '/dev/null'))
    (HERE / name).write_text(''.join(lines))
    return changes


def main():
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD',
                                      '--', 'ssd/ssd'], cwd=ROOT, text=True).splitlines()
    head = {p: subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT, text=True)
            for p in tracked if p.endswith('.py')}
    frozen = checkout_runtime(FROZEN)
    current = checkout_runtime(ROOT)
    approved = {k: v for k, v in frozen.items() if k != EXPERIMENTAL}
    production = {k: v for k, v in current.items() if k != EXPERIMENTAL}
    stages = [dict(name='01_approved_extension', patch='APPROVED_EXTENSION.patch',
                   files=patch(head, approved, 'APPROVED_EXTENSION.patch')),
              dict(name='02_followup_and_review_fixes', patch='FOLLOWUP_RUNTIME.patch',
                   files=patch(approved, production, 'FOLLOWUP_RUNTIME.patch')),
              dict(name='03_review_tools_tests', patch='REVIEW_RUNTIME.patch',
                   files=patch(production, current, 'REVIEW_RUNTIME.patch'))]
    # Materialize and apply all layers away from the working tree and index.
    with tempfile.TemporaryDirectory(prefix='duet-port-check-') as tmp:
        temp = Path(tmp)
        for path, text in head.items():
            dest = temp / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
        for stage in stages:
            subprocess.run(['git', 'apply', str(HERE / stage['patch'])],
                           cwd=temp, check=True)
        if checkout_runtime(temp) != current:
            raise AssertionError('Review layers do not reproduce the current runtime')

    dirty = set(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT,
                                        text=True).splitlines())
    dirty.update(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'],
                                        cwd=ROOT, text=True).splitlines())
    tools = sorted(p for p in dirty if p.startswith(('ssd/bench/', 'ssd/tests/')))
    stages[2]['files'] += tools
    stages[2]['files'].append('results/mlsys_ver01/extensions/check_suites.py')
    stages.append(dict(name='04_reports_and_evidence', files=sorted(
        p for p in dirty if p.startswith('results/') and p not in stages[2]['files'])))
    manifest = dict(base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                 text=True).strip(),
                    stages=stages, runtime_layers_reproduce_current=True,
                    runtime_hashes={p: sha(ROOT / p) for p in sorted(current)},
                    notes=['No git index or branch was modified. These are proposed commit layers.',
                           'The earlier ADOPTED_RUNTIME.patch is historical evidence; this followup patch includes the new precommit fixes.',
                           'The optional packed target helper is separated from default production runtime.',
                           'Some files occur in both production layers because hunks must be committed in sequence; git add of whole files would collapse that boundary.'])
    (HERE / 'COMMIT_PLAN.json').write_text(json.dumps(manifest, indent=2))
    with tarfile.open(HERE / 'CHECK_LOGS.tar.gz', 'w:gz') as tar:
        for file in sorted(HERE.rglob('*')):
            if file.is_file() and (file.suffix == '.log' or file.parent.name in ('final_suites', 'pre_policy_suites')):
                tar.add(file, arcname=str(file.relative_to(HERE)))
    with tarfile.open(HERE / 'FINAL_SOURCE.tar.gz', 'w:gz') as tar:
        for path in sorted(current):
            tar.add(ROOT / path, arcname=path)
        for path in tools + ['results/mlsys_ver01/extensions/check_suites.py',
                             'results/mlsys_ver01/REPORT.md']:
            tar.add(ROOT / path, arcname=path)
    artifacts = [p for p in HERE.glob('*') if p.is_file() and p.name != 'ARTIFACTS.json']
    (HERE / 'ARTIFACTS.json').write_text(json.dumps({p.name: dict(bytes=p.stat().st_size,
                    sha256=sha(p)) for p in artifacts}, indent=2))
    print('Layered patches reproduce final runtime; artifacts saved.', flush=True)


if __name__ == '__main__':
    main()
