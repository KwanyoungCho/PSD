"""Summarize the final-runtime, one-factor TP2 online comparison."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = Path('/data/chokwans99/duet-precommit-20261014/tp2_full')
sys.path.insert(0, str(ROOT / 'ssd/bench'))
from mlsys_tree_results import summarize, compare


def main():
    reports = {name: json.loads((RAW / (name + '.json')).read_text())
               for name in ('base', 'auto')}
    if any(j['status'] != 'complete' for j in reports.values()):
        raise RuntimeError('Both arms must be complete')
    checks = json.loads((HERE / 'pre_policy_suites/CHECKS_PROVENANCE.json').read_text())
    check_runtime = {k: v for k, v in checks['source_hashes'].items()
                     if k.startswith('ssd/ssd/')}
    for name, j in reports.items():
        if j['source_hashes'] != check_runtime:
            raise AssertionError(f'{name} runtime differs from full regression runtime')
    if reports['base']['args'] != reports['auto']['args']:
        # Output destination is the only argument intentionally different.
        for k in reports['base']['args']:
            if k != 'output' and reports['base']['args'][k] != reports['auto']['args'][k]:
                raise AssertionError(f'Unexpected parameter change: {k}')
    offset = time.time() - time.monotonic()
    telemetry = {}
    for name, j in reports.items():
        start = min(s['audit_start_ns'] for s in j['steps']) / 1e9 + offset
        end = max(s['audit_end_ns'] for s in j['steps']) / 1e9 + offset
        readings = [json.loads(l) for l in (RAW / (name + '.telemetry.jsonl')).read_text().splitlines()]
        window = [x for x in readings if start <= x['time'] <= end]
        gpu = {}
        for row in window:
            for line in row['gpu'].splitlines():
                index, memory, util, clock, power, temp = map(float, line.split(','))
                entry = gpu.setdefault(str(int(index)), dict(clocks=[], temperatures=[], memory=[]))
                entry['temperatures'].append(temp)
                entry['memory'].append(memory)
                if util >= 20:
                    entry['clocks'].append(clock)
        telemetry[name] = dict(samples=len(window), external_pids=sorted({p for x in readings for p in x['external_pids']}),
            gpus={k: dict(active_mean_clock_mhz=float(np.mean(v['clocks'])) if v['clocks'] else None,
                          max_temperature=max(v['temperatures']), max_memory_mib=max(v['memory']))
                  for k, v in gpu.items()})
    out = dict(arms={k: summarize(j) for k, j in reports.items()},
               comparison=compare(reports['base'], reports['auto']),
               resolved={k: j['resolved_tree_runtime'] for k, j in reports.items()},
               same_runtime_as_pre_policy_regression=True, telemetry=telemetry,
               clock_alignment_offset_seconds=offset,
               checks=dict(suites=len(json.loads((HERE / 'pre_policy_suites/CHECKS.json').read_text())),
                           tests=sum(x['tests'] for x in json.loads((HERE / 'pre_policy_suites/CHECKS.json').read_text()))),
               raw_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in RAW.rglob('*') if p.is_file() and p.suffix != '.tmp'})
    (HERE / 'TP2_FULL_RESULTS.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('comparison', 'resolved', 'telemetry')}, indent=2))


if __name__ == '__main__':
    main()
