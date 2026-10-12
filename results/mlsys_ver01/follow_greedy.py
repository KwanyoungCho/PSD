"""After the owned greedy campaign, diagnose every first AR/tree difference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
deadline = time.time() + 7200
while True:
    f = R / 'greedy/campaign.json'
    rows = json.loads(f.read_text()) if f.exists() else []
    if any(v['status'] != 'complete' or v['exit_code'] for v in rows):
        raise RuntimeError('Greedy prerequisite failed')
    if len(rows) == len(json.loads((R / 'greedy_plan.json').read_text())):
        break
    if time.time() > deadline:
        raise TimeoutError('Greedy prerequisite not completed')
    time.sleep(5)

tree = json.loads((R / 'greedy/tree.json').read_text())
ar = json.loads((R / 'greedy/ar.json').read_text())
questions = json.loads((R / 'questions.json').read_text())
focus, subset = [], []
for i, (a, b) in enumerate(zip(tree['outputs'], ar['outputs'])):
    if a == b:
        continue
    pos = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    if pos >= min(len(a), len(b)):
        raise AssertionError('Length-only mismatch requires separate EOS/cap diagnosis')
    prefix = tree['prompt_ids'][i] + a[:pos]
    key = hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
    focus.append(dict(index=i, offset=pos, key=key, tree_token=a[pos], ar_token=b[pos]))
    subset.append(questions[i])
(R / 'greedy_focus.json').write_text(json.dumps(focus, indent=2))
(R / 'greedy_questions.json').write_text(json.dumps(subset, indent=2))
print('First divergences:', len(focus), flush=True)
if not focus:
    (R / 'GREEDY_DIAGNOSIS.json').write_text(json.dumps(dict(mismatches=0, status='complete')))
    sys.exit(0)

plan = []
for mode in ('ar', 'tree'):
    baseline = mode == 'ar'
    plan.append(dict(name=mode, arm='base', draft_seed=231,
                     bench='mlsys_ver01_greedy_trace.py',
                     args=['--mode', mode, '--prompts', str(R / 'greedy_questions.json'),
                           '--output-tokens', '96', '--warm-prompts', '0',
                           '--temperature', '0', '--seed', '10801'],
                     env=dict(PAPER_CPUSET='0-7,32-39',
                              PAPER_CHECKOUT='/home/chokwans99/PSD-paper-review' if baseline else str(ROOT),
                              PAPER_BASELINE='1' if baseline else '0',
                              GREEDY_TRACE_FOCUS=str(R / 'greedy_focus.json'),
                              GREEDY_TRACE_DIR=str(R / 'greedy_trace' / mode))))
(R / 'greedy_trace_plan.json').write_text(json.dumps(plan, indent=2))
# The completed controller may still be releasing its process group.
time.sleep(3)
subprocess.run([sys.executable, str(R / 'run_pair.py'), '--plan', str(R / 'greedy_trace_plan.json'),
                '--gpus', '0,1', '--port', '30110', '--directory', str(R / 'greedy_trace_runs')],
               cwd=ROOT, check=True)

import torch
records = {}
for mode in ('ar', 'tree'):
    p = R / 'greedy_trace' / mode / 'records.jsonl'
    records[mode] = {x['key']: x for line in p.read_text().splitlines() if (x := json.loads(line))}
diagnostics = []
for f in focus:
    key = f['key']
    entry = dict(f)
    values = {}
    for mode in ('ar', 'tree'):
        if key not in records[mode]:
            raise AssertionError(('First-divergence prefix was not captured', mode, f))
        values[mode] = torch.load(R / 'greedy_trace' / mode / (key + '.pt'), weights_only=True)
        x = values[mode]
        top = x.topk(2).values
        entry[mode] = dict(argmax=int(x.argmax()), top2_gap=float(top[0]-top[1]),
                           shape=records[mode][key]['shape'],
                           tree_step=records[mode][key]['tree_meta'] is not None)
        if entry[mode]['argmax'] != f[mode + '_token']:
            raise AssertionError(('Emission did not match captured target argmax', mode, entry))
    entry['max_logit_difference'] = float((values['tree']-values['ar']).abs().max())
    diagnostics.append(entry)
reproduces = {}
for mode, full in [('ar', ar), ('tree', tree)]:
    d = json.loads((R / 'greedy_trace_runs' / (mode + '.json')).read_text())
    reproduces[mode] = d['outputs'] == [full['outputs'][f['index']] for f in focus]
    if not reproduces[mode]:
        raise AssertionError(('Diagnostic did not reproduce measured output', mode))
result = dict(status='complete', mismatches=len(focus),
              captured={k: len(v) for k, v in records.items()}, all_emissions_are_own_target_argmax=True,
              reproduces_full_outputs=reproduces, rows=diagnostics)
(R / 'GREEDY_DIAGNOSIS.json').write_text(json.dumps(result, indent=2) + '\n')
print('All first divergences captured; both paths emitted their own argmax.', flush=True)
