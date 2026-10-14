"""Preserve all completed/ineligible observations and the tested source."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((HERE/'checks/CHECKS_PROVENANCE.json').read_text())
assert proof['source_unchanged']
assert all(sha(ROOT/k)==v for k,v in proof['source_hashes'].items())
result=json.loads((HERE/'RESULTS.json').read_text())
assert result['status']=='complete'
with tarfile.open(HERE/'TESTED_SOURCE.tar.gz','w:gz') as archive:
    for k in sorted(proof['source_hashes']):archive.add(ROOT/k,arcname=k)
raw_paths=[Path('/data/chokwans99')/name for name in (
    'ver02-payload-component1-20261014.json','ver02-payload-component1-20261014.log',
    'ver02-payload-component2-20261014','ver02-payload-component3-20261014',
    'ver02-payload-full-l2-20261014','ver02-payload-full-l2-retry-20261014',
    'ver02-payload-nova-l2-20261014','ver02-payload-nova-l3-20261014',
    'ver02-payload-tests-20261014.log','ver02-remaining-checks-20261014.log')]
raw_hashes={}
with tarfile.open(HERE/'RAW_REVIEW.tar.gz','w:gz') as archive:
    for p in raw_paths:
        assert p.exists(),p
        files=sorted(p.rglob('*')) if p.is_dir() else [p]
        for f in files:
            if not f.is_file():continue
            name=str(f.relative_to(p.parent))
            raw_hashes[name]=sha(f)
            archive.add(f,arcname=name)
(HERE/'RAW_MANIFEST.json').write_text(json.dumps(raw_hashes,indent=2))
manifest=dict(base_release='0d5374289b8112db53c8f82ea7eafadc0c30a696',
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    tested_source_hashes=proof['source_hashes'],
    artifacts={str(f.relative_to(HERE)):sha(f) for f in sorted(HERE.rglob('*'))
               if f.is_file() and f.name!='MANIFEST.json'})
(HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('Saved',len(raw_hashes),'raw files;',len(manifest['artifacts']),'artifacts')
