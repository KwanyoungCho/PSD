"""Trace first AR/tree differences at identical prefixes; not a speed test.

The observer is installed only in this diagnostic process. Production code,
model configuration, actual logits and sampling remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def key(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def outputs(record):
    return [tokens for group in record['groups'] for tokens in group['outputs']]


def worker(reference_path, focus_path, directory):
    sys.path.insert(0, str(ROOT / 'ssd'))
    import torch
    from ssd.engine.model_runner import ModelRunner
    from ssd.engine.helpers.p2_tree import parse_tree_ints
    import probe
    reference = read(reference_path)
    focus = {key(tokens): tokens for tokens in read(focus_path)}
    directory.mkdir(parents=True, exist_ok=False)
    old_run, old_model = ModelRunner.run, ModelRunner.run_model

    def run(self, seqs, *args, **kwargs):
        self._diagnostic_tokens = [list(s.token_ids) for s in seqs]
        return old_run(self, seqs, *args, **kwargs)

    def model(self, input_ids, positions, is_prefill, last_only=True, *args, **kwargs):
        logits = old_model(self, input_ids, positions, is_prefill, last_only, *args, **kwargs)
        if (self.is_draft or self.rank != 0 or not isinstance(logits, torch.Tensor)
                or not hasattr(self, '_diagnostic_tokens')):
            return logits
        if len(self._diagnostic_tokens) != 1:
            raise ValueError('This observer supports B=1 only')
        tokens = self._diagnostic_tokens[0]
        meta = getattr(self, '_duet_tree_meta', None)
        if is_prefill or last_only:
            prefixes, rows = [tokens], [logits.shape[0] - 1]
        else:
            ids, pos = input_ids.tolist(), positions.tolist()
            base = tokens[:pos[0]]
            if meta is not None:
                tree = parse_tree_ints(meta, (len(meta) - 3) // 4)
                prefixes = [base + [ids[0]]]
                for j in range(tree['valid']):
                    prefixes.append(prefixes[tree['parent_local'][j] + 1] + [tree['tok'][j]])
            else:
                prefixes = [base + ids[:i + 1] for i in range(len(ids))]
            rows = list(range(len(prefixes)))
        flat = logits.reshape(-1, logits.shape[-1])
        for prefix, row in zip(prefixes, rows):
            k = key(prefix)
            destination = directory / (k + '.pt')
            if k not in focus or destination.exists():
                continue
            if prefix != focus[k]:
                raise ValueError('Prefix identity mismatch')
            value = flat[row].detach().float().cpu().clone()
            torch.save(value, destination)
            vals, ids = value.topk(16)
            record = dict(key=k, prefix=prefix, row=row, shape=list(logits.shape),
                          tree_step=meta is not None, argmax=int(value.argmax()),
                          top_ids=ids.tolist(), top_values=vals.tolist())
            with (directory / 'logits.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
        return logits

    ModelRunner.run, ModelRunner.run_model = run, model
    opts = reference['args']
    sys.argv = [str(HERE / 'probe.py')]
    for name in ('target', 'draft', 'mode', 'expansion', 'normalization', 'temperature', 'batch'):
        sys.argv += ['--' + name, str(opts[name])]
    sys.argv += ['--output', str(directory / 'run.json')]
    probe.main()
    rerun = read(directory / 'run.json')
    if (rerun['inputs'] != reference['inputs'] or outputs(rerun) != outputs(reference)
            or not rerun['source_unchanged']):
        raise AssertionError('Observer did not reproduce the uninstrumented run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--focus', type=Path)
    parser.add_argument('--tree-after-checks', type=Path,
                        help='Wait for this regression directory to finish before using the draft GPU')
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.focus, args.output_dir)
    models = args.models_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = read(models / 'status.json')
    if any(r.get('returncode') != 0 for r in original['runs']):
        raise AssertionError('All original native runs must have finished successfully')
    report = dict(status='running', started=time.time(), differences=[], runs=[])

    def save():
        (out / 'DIAGNOSIS.json').write_text(json.dumps(report, indent=2))

    for model_name in ('layerskip', 'base70b'):
        ar_name = model_name + '_ar_last_t0_full_b1'
        ar = read(models / (ar_name + '.json'))
        names = [model_name + '_tree_last_t0_full_b1']
        if model_name == 'layerskip':
            names += [model_name + '_tree_frontier_t0_topm_b1']
        focus, needed = {}, set()
        for name in names:
            tree = read(models / (name + '.json'))
            if tree['inputs'] != ar['inputs']:
                raise AssertionError('Different input IDs')
            for i, (a, b) in enumerate(zip(outputs(ar), outputs(tree))):
                if a == b:
                    continue
                pos = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                if pos >= min(len(a), len(b)):
                    raise AssertionError('Length-only mismatch needs EOS/cap diagnosis')
                tokens = ar['inputs'][i] + a[:pos]
                focus[key(tokens)] = tokens
                report['differences'].append(dict(model=model_name, name=name, ar_name=ar_name,
                    prompt=i, offset=pos, key=key(tokens), ar_token=a[pos], tree_token=b[pos]))
                needed.update((name, ar_name))
        path = out / (model_name + '_focus.json')
        path.write_text(json.dumps(list(focus.values()), indent=2))
        for name in sorted(needed):
            if '_tree_' in name and args.tree_after_checks:
                marker = args.tree_after_checks / 'CHECKS_PROVENANCE.json'
                deadline = time.time() + 1200
                while not marker.exists():
                    if time.time() > deadline:
                        raise TimeoutError('Regression did not release the draft GPU')
                    time.sleep(2)
                if (not read(marker)['source_unchanged'] or any(
                        x['returncode'] for x in read(args.tree_after_checks / 'CHECKS.json'))):
                    raise AssertionError('Regression prerequisite failed')
            previous = next(r for r in original['runs'] if r['name'] == name)
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'ssd'), SSD_HF_CACHE='/tmp',
                       SSD_DATASET_DIR='/tmp', SSD_CUDA_ARCH='12.0', SSD_DEBUG_TOKENS='0',
                       SSD_TREE_EXEC_WARMUP='all', SSD_DIST_PORT='29895', OMP_NUM_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=previous['gpus'])
            for variable in ('SSD_TREE_EXPANSION_POLICY', 'SSD_TREE_LEGACY_CALIBRATION', 'LD_PRELOAD'):
                env.pop(variable, None)
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                       str(models / (name + '.json')), '--focus', str(path),
                       '--output-dir', str(out / name)]
            row = dict(name=name, command=command, started=time.time())
            report['runs'].append(row); save()
            with (out / (name + '.log')).open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = process.pid; save()
                try:
                    code = process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    code = 124
            row.update(returncode=code, ended=time.time()); save()
            if code:
                report['status'] = 'failed'; save()
                raise RuntimeError('Diagnostic failed: ' + name)
    import torch
    for row in report['differences']:
        values = {}
        for mode, name in (('ar', row['ar_name']), ('tree', row['name'])):
            values[mode] = torch.load(out / name / (row['key'] + '.pt'), weights_only=True)
            x = values[mode]
            top = x.topk(2).values
            row[mode] = dict(argmax=int(x.argmax()), top2_gap=float(top[0] - top[1]),
                            competing_logits=[float(x[row['ar_token']]), float(x[row['tree_token']])])
            if row[mode]['argmax'] != row[mode + '_token']:
                report['status'] = 'failed'; save()
                raise AssertionError('Emitted token differs from own target argmax')
        delta = (values['tree'] - values['ar']).abs()
        row.update(max_logit_difference=float(delta.max()), mean_logit_difference=float(delta.mean()))
    report.update(status='complete', ended=time.time(), all_emissions_are_own_target_argmax=True,
                  strict_token_parity_passed=False, observer_reproduced_original_outputs=True)
    save()
    print(json.dumps(report))


if __name__ == '__main__':
    main()
