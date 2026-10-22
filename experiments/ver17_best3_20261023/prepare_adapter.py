"""Reconstruct the frozen measurement helpers without copying the engine.

Existing files are verified, never overwritten. Run before a new campaign.
"""
import hashlib
import gzip
import json
from pathlib import Path
import subprocess

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]


def main():
    manifest = json.loads((CODE / 'ADAPTER_MANIFEST.json').read_text())
    data = gzip.decompress((CODE / 'INPUTS_FULL.json.gz').read_bytes())
    assert hashlib.sha256(data).hexdigest() == manifest['inputs']['sha256']
    dest = CODE / 'INPUTS_FULL.json'
    if dest.exists():
        assert dest.read_bytes() == data, f'Existing inputs differ: {dest}'
    else:
        dest.write_bytes(data)
    for relative, entry in manifest['files'].items():
        if entry['source'] == 'override':
            data = (CODE / 'adapter_overrides' / relative).read_bytes()
        else:
            data = subprocess.check_output(
                ['git', 'show', f"{manifest['commit']}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], relative
        dest = CODE / 'adapter' / relative
        if dest.exists():
            assert dest.read_bytes() == data, f'Existing adapter differs: {dest}'
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    print(f"Verified exact input bytes and {len(manifest['files'])} frozen measurement helpers")


if __name__ == '__main__':
    main()
