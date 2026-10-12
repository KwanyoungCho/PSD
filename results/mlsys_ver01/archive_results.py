"""Keep raw measured runs, failed-test evidence and traces in one portable archive."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
folders = ('tree_r0', 'tree_r1', 'chain', 'greedy', 'profiles', 'greedy_trace', 'greedy_trace_runs')
files = sorted({p for name in folders for p in (R/name).rglob('*') if p.is_file()} |
               {p for p in R.glob('*.log') if p.name != 'archive.log'})
rows = [dict(path=str(p.relative_to(R)), size=p.stat().st_size,
             sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
archive = R/'RAW_RESULTS.tar.gz'
with tarfile.open(archive, 'w:gz') as out:
    for p in files:
        out.add(p, arcname=str(p.relative_to(R)))
with tarfile.open(archive) as inp:
    assert len(inp.getmembers()) == len(rows)
    for row in rows:
        content = inp.extractfile(row['path']).read()
        assert hashlib.sha256(content).hexdigest() == row['sha256']
(R/'RAW_MANIFEST.json').write_text(json.dumps(dict(archive=archive.name,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=rows,
    all_members_read_back_and_checked=True), indent=2)+'\n')
patch = subprocess.check_output(['git','diff','a82f7d2','--','ssd/ssd','ssd/bench/bench.py'],
                                 cwd=ROOT, text=True)
(R/'RUNTIME.patch').write_text(patch)
print('Verified archive', archive.stat().st_size, 'bytes;', len(files), 'files')
