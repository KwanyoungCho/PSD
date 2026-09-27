"""Finish reports only after all six frozen jobs succeed; optionally watch progress."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = str(ROOT / 'ssd/.venv/bin/python')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait', action='store_true')
    args = ap.parse_args()
    plan = json.loads((HERE / 'plan.json').read_text())
    while True:
        progress = []
        for job in plan['jobs']:
            folder = HERE / 'runs' / f"s{job['seed']}_{job['policy']}"
            completion = folder / 'completion.json'
            status = folder / 'status.json'
            item = dict(**job, state='pending', turns=0)
            if status.exists():
                item.update(state='running', turns=json.loads(status.read_text())['completed_turns'])
            elif folder.exists():
                item['state'] = 'initializing'
            if completion.exists():
                outcome = json.loads(completion.read_text())
                item['state'] = 'complete' if outcome['exit_code'] == 0 else 'failed'
            progress.append(item)
        document = dict(updated_utc=datetime.now(timezone.utc).isoformat(), jobs=progress,
                        completed_turns=sum(p['turns'] for p in progress), total_turns=3360)
        temporary = HERE / 'campaign_status.tmp'
        temporary.write_text(json.dumps(document, indent=2))
        temporary.replace(HERE / 'campaign_status.json')
        if any(p['state'] == 'failed' for p in progress):
            raise RuntimeError('A generation job failed; original files were preserved. Inspect its run.log.')
        if all(p['state'] == 'complete' for p in progress):
            break
        if not args.wait:
            raise RuntimeError('Campaign is incomplete; use --wait to finish automatically when ready.')
        time.sleep(15)
    # Generation code must not change during the comparison. Analysis and
    # reporting files are explicitly separate and can receive audit fixes.
    snapshot = json.loads((HERE / 'source_snapshot.json').read_text())
    protected = {k: v for k, v in snapshot['files'].items()
                 if not k.startswith('results/duet_tree_al_full/')
                 or Path(k).name in ('runtime.py', 'run_full.py', 'plan.json', 'questions.json', 'upstream_question.jsonl')}
    for relative, expected in protected.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Protected source/data changed: ' + relative)
    for script in ('validate.py', 'analyze.py', 'make_report.py'):
        result = subprocess.run([PYTHON, str(HERE / script)], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (HERE / (script.removesuffix('.py') + '_stdout.log')).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'{script} failed; inspect its stdout log')
        print('completed', script, flush=True)
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.json')) + list(HERE.glob('*.md'))
    paths += [HERE / 'NUMBERS.txt', HERE / 'task_results.csv', HERE / 'comparison.png', HERE / 'comparison.pdf']
    for job in plan['jobs']:
        folder = HERE / 'runs' / f"s{job['seed']}_{job['policy']}"
        paths += list(folder.glob('*.json')) + [folder / 'records.jsonl']
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        protected_sources_unchanged=True,
        files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(set(paths)) if p.name not in ('final_manifest.json', 'finalized.json')})
    (HERE / 'final_manifest.json').write_text(json.dumps(manifest, indent=2))
    (HERE / 'finalized.json').write_text(json.dumps(dict(complete=True,
        completed_utc=manifest['completed_utc'], questions_per_run=480, turns_per_run=560,
        runs=6, total_turns=3360, report='REPORT.md'), indent=2))
    print('Full experiment and report complete.', flush=True)


if __name__ == '__main__':
    main()
