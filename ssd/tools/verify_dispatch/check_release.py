"""CPU-only current release runtime integrity check; never import torch or run a GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    manifest = json.loads((ROOT / 'results/mlsys_ver110/RUNTIME_MANIFEST.json').read_text())
    actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / 'ssd/ssd').rglob('*.py'))}
    if actual != manifest['sha256']:
        expected = manifest['sha256']
        changed = sorted(k for k in actual.keys() | expected.keys()
                         if actual.get(k) != expected.get(k))
        raise SystemExit('Runtime integrity FAILED: ' + ', '.join(changed))
    for name, digest in manifest.get('tools_sha256', {}).items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit('Measurement/tool integrity FAILED: ' + name)
    source = manifest.get('input')
    if source and hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() != source['sha256']:
        raise SystemExit('Input integrity FAILED: ' + source['path'])
    print(json.dumps(dict(state='passed', release=manifest['release'],
                          runtime_files=len(actual), gpu_work=False)))


if __name__ == '__main__':
    main()
