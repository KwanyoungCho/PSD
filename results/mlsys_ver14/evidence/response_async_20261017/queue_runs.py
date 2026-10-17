"""One GPU pair/triple at a time, frozen sources, exact output/event parity."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data2/chokwans99/response-async-20261017')


def signature(data):
    return [(e.get('source'), e.get('cache_hit'), e.get('valid_k'), e.get('accepted_len'))
            for s in data['steps'] for e in s['events']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['7b', '13b'], required=True)
    p.add_argument('--schedule', type=Path, required=True)
    args = p.parse_args()
    schedule = json.loads(args.schedule.read_text())
    status_path = ROOT / (args.schedule.stem + '_' + args.model + '.json')
    completed = []
    for stage in schedule:
        for check in stage.get('checks', []):
            path = ROOT / check / 'CHECKS.json'
            while not path.exists():
                time.sleep(3)
            if json.loads(path.read_text())['status'] != 'complete':
                raise RuntimeError('Runtime checks failed: ' + str(path))
        dependency = stage.get('wait')
        if dependency:
            wait = ROOT / 'runs' / dependency.format(model=args.model) / 'status.json'
            while not wait.exists():
                time.sleep(3)
            if json.loads(wait.read_text())['status'] != 'complete':
                raise RuntimeError('Dependency failed: ' + str(wait))
        tag = stage['tag'].format(model=args.model)
        folder = ROOT / 'runs' / tag
        if not folder.exists():
            argv = [sys.executable, str(ROOT / 'remote_run.py'), '--variant', stage['variant'],
                    '--model', args.model, '--tag', tag, '--passes', str(stage['passes'])]
            if stage.get('profile'):
                argv.append('--profile')
            subprocess.run(argv, check=True)
        status = json.loads((folder / 'status.json').read_text())
        if status['status'] != 'complete':
            raise RuntimeError(status)
        if 'reference' in stage:
            ref = ROOT / 'runs' / stage['reference'].format(model=args.model) / 'run.json'
            a = json.loads(ref.read_text()); b = json.loads((folder / 'run.json').read_text())
            parity = dict(outputs_equal=a['outputs'] == b['outputs'],
                          events_equal=signature(a) == signature(b),
                          outputs=len(b['outputs']), events=len(signature(b)))
            (folder / 'PARITY.json').write_text(json.dumps(parity, indent=2) + '\n')
            if not parity['outputs_equal'] or not parity['events_equal']:
                raise RuntimeError((tag, parity))
        completed.append(tag)
        status_path.write_text(json.dumps(dict(status='running', completed=completed,
                                               schedule=schedule), indent=2) + '\n')
        print('COMPLETE', tag, flush=True)
    status_path.write_text(json.dumps(dict(status='complete', completed=completed,
                                           schedule=schedule), indent=2) + '\n')


if __name__ == '__main__':
    main()
