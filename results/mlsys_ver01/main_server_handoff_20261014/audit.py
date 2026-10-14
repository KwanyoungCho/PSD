"""Read-only CPU audit of the supplied Blackwell zero-decode handover.

This does not validate the remote report.py or its unavailable raw JSONL.
No model is loaded, no GPU is used, and old evidence is not overwritten.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCHEDULER = 'ssd/ssd/engine/scheduler.py'
HELPERS = 'ssd/ssd/utils/async_helpers/async_spec_helpers.py'


def node(source, name):
    return next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def at(rev, path):
    return subprocess.check_output(['git', 'show', f'{rev}:{path}'],
                                   cwd=ROOT, text=True)


def module(path):
    # Newer benchmark helpers have sibling imports; keep this historical
    # source audit runnable without importing the model engine.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def main():
    out = dict(scope='Local source/CPU checks and three local raw files only; '
               'remote code, 9 tests, 15 runs and 27-job completion not verified',
               engine_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                  cwd=ROOT, text=True).strip(),
               source_checks={}, boundary_cases=[], local_raw=[])
    for path, name in [(SCHEDULER, '_model_length_block_reason'),
                       (HELPERS, 'compute_megaspec_lookahead')]:
        src = [at('a82f7d2', path), at('f12d0f1', path), (ROOT / path).read_text()]
        dumps = [ast.dump(node(s, name), include_attributes=False) for s in src]
        assert dumps[0] == dumps[1] == dumps[2], name
        out['source_checks'][name] = dict(paper_pushed_current_equal=True,
            ast_sha256=hashlib.sha256(dumps[0].encode()).hexdigest())

    guard_node = node((ROOT / SCHEDULER).read_text(), '_model_length_block_reason')
    ns = {}
    exec('from __future__ import annotations\n' + ast.unparse(guard_node), ns)
    guard = ns['_model_length_block_reason']
    # The 325-slot value is supplied by the remote handover, not reconstructed
    # from the unavailable remote plan.json.
    for prompt, target, draft, expected in [
            (1734, 13, 325, 'draft'), (1723, 13, 325, None),
            (1724, 13, 325, 'draft'), (2040, 13, 1, 'target')]:
        reason = guard(SimpleNamespace(max_model_len=2048, speculate=True),
                       SimpleNamespace(num_tokens=prompt), target, draft)
        assert (None if reason is None else reason.split()[0]) == expected
        out['boundary_cases'].append(dict(prompt=prompt, target=target,
            draft=draft, reason=reason, passed=True))

    perf = module(ROOT / 'ssd/ssd/utils/performance.py').performance_summary({}, 1)
    measured = module(ROOT / 'ssd/bench/mlsys_measurement.py').summarize([], 1, [[]])
    assert all(perf[k] is None for k in ('tps', 'al', 'cache_hit'))
    assert all(measured['summary'][k] is None
               for k in ('tps', 'al', 'hit', 'step_ms', 'verify_ms'))
    out['empty_aggregation'] = dict(engine=perf, benchmark=measured,
        note='Empty-population arithmetic only; not validation of a termination reason.')

    raw_root = Path('/data/chokwans99/duet-precommit-20261014')
    for rel in ('tp2_full/base.json', 'tp2_full/auto.json',
                'tp2_final/final_default.json'):
        path = raw_root / rel
        data = path.read_bytes()
        j = json.loads(data)
        row = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                   status=j['status'], output_records=len(j['outputs']),
                   empty_outputs=sum(not x for x in j['outputs']),
                   max_prompt_tokens=max(map(len, j['prompt_ids'])),
                   measured_groups=len(j['groups']),
                   empty_decode_groups=sum(not g['steps'] for g in j['groups']))
        assert row['status'] == 'complete' and row['empty_outputs'] == 0
        assert row['empty_decode_groups'] == 0
        out['local_raw'].append(row)

    out['historical_local_reporting_gap'] = (
        'mlsys_tree_followup.py computes an intermediate TPS with an unguarded '
        'division before finalize_measurement. An entirely empty eligible '
        'population would fail there; a single zero-decode request does not '
        'cause that when other valid steps remain. The local schema also lacks '
        'per-request finish reasons needed for the remote strict classification. '
        'No engine or benchmark changes made by this audit. '
        'MLSys-ver0.2 implements the reporting fix; see results/mlsys_ver02/REPORT.md. '
        'The original AUDIT.json is evidence of the earlier review state.')
    (HERE / 'AUDIT.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(source_parity=True, boundary_cases=4,
        empty_aggregators=2, raw_files=out['local_raw']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
