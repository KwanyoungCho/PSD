"""Verify the final opt-in policy against its full regression snapshot."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = Path('/data/chokwans99/duet-precommit-20261014')


def main():
    old = json.loads((RAW / 'tp2_full/base.json').read_text())
    final = json.loads((RAW / 'tp2_final/final_default.json').read_text())
    if final['status'] != 'complete':
        raise RuntimeError('Final full-model check is unfinished')
    prov = json.loads((HERE / 'final_suites/CHECKS_PROVENANCE.json').read_text())
    expected = {p: h for p, h in prov['source_hashes'].items() if p.startswith('ssd/ssd/')}
    current = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT / 'ssd/ssd').rglob('*.py')}
    if not (final['source_hashes'] == expected == current and prov['source_unchanged']):
        raise AssertionError('Final run, regression and working runtime differ')
    modes = final['resolved_tree_runtime']
    if modes['exit_replica'] or modes['proxy_stream'] or modes['replica_source'] != 'default':
        raise AssertionError('Final unset-env default did not select OFF')
    checks = json.loads((HERE / 'final_suites/CHECKS.json').read_text())
    if any(r['returncode'] for r in checks):
        raise AssertionError('Full regression failed')
    out = dict(status='complete', source_matches=True,
               same_all_outputs=old['outputs'] == final['outputs'],
               equal_output_sequences=sum(x == y for x, y in zip(old['outputs'], final['outputs'])),
               output_sequences=len(final['outputs']), resolved=modes,
               baseline_summary=old['summary'], final_summary=final['summary'],
               preparation=final['measurement']['preparation_all_decode'],
               changed_runtime_files=[p for p in expected if expected[p] != old['source_hashes'][p]],
               scope='Functional full-model check only; some regression tests ran concurrently on separate GPU7. Do not claim latency improvement from this run.')
    (HERE / 'FINAL_DEFAULT_CHECK.json').write_text(json.dumps(out, indent=2))
    for name in ('CHECKS.json', 'CHECKS_PROVENANCE.json'):
        shutil.copy2(HERE / 'final_suites' / name, HERE.parent / 'extensions' / name)
    hashes = {}
    with tarfile.open(HERE / 'RAW_TP2.tar.gz', 'w:gz') as tar:
        for f in sorted(RAW.rglob('*')):
            if f.is_file() and f.suffix != '.tmp':
                rel = str(f.relative_to(RAW))
                hashes[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
                tar.add(f, arcname=rel)
    (HERE / 'RAW_MANIFEST.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
