"""Deterministic raw-result gzip archives with verified restore hashes."""
import argparse,gzip,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');a=p.parse_args()
manifest=R/'ARCHIVE_MANIFEST.json'
def sha(b):return hashlib.sha256(b).hexdigest()
if a.restore:
 for row in json.loads(manifest.read_text()):
  packed=(R/row['archive']).read_bytes()
  if sha(packed)!=row['archive_sha256']:raise ValueError(row['archive'])
  data=gzip.decompress(packed)
  if sha(data)!=row['raw_sha256']:raise ValueError(row['raw'])
  raw=R/row['raw'];raw.parent.mkdir(parents=True,exist_ok=True)
  if raw.exists() and sha(raw.read_bytes())!=row['raw_sha256']:raise FileExistsError(raw)
  raw.write_bytes(data)
else:
 rows=[]
 files=set()
 for plan in R.glob('*/plan.json'):
  for job in json.loads(plan.read_text()):
   for suffix in ('.json','.log','.telemetry.jsonl'):
    path=plan.with_name(job['name']+suffix)
    if not path.exists():raise FileNotFoundError(path)
    files.add(path)
   data=json.loads(plan.with_name(job['name']+'.json').read_text())
   if data['status']!='complete':raise ValueError('Incomplete model result '+job['name'])
 files.update((R/'profiles').rglob('*.json') if (R/'profiles').exists() else [])
 files.update(p for p in (R/'failed_attempts').rglob('*') if p.is_file())
 files.update(R.glob('smoke*.json'));files.update(R.glob('smoke*.log'))
 for raw in sorted(files):
  data=raw.read_bytes();rel=raw.relative_to(R);dest=R/'archives'/str(rel).replace('/','__')
  dest=dest.with_name(dest.name+'.gz');dest.parent.mkdir(parents=True,exist_ok=True)
  packed=gzip.compress(data,compresslevel=6,mtime=0)
  if gzip.decompress(packed)!=data:raise AssertionError('Compression validation failed')
  dest.write_bytes(packed)
  rows.append(dict(raw=str(rel),archive=str(dest.relative_to(R)),raw_bytes=len(data),archive_bytes=len(packed),
                   raw_sha256=sha(data),archive_sha256=sha(packed)))
 manifest.write_text(json.dumps(rows,indent=2))
 print('Archived files',len(rows),'raw MB',sum(x['raw_bytes'] for x in rows)/1e6,'gzip MB',sum(x['archive_bytes'] for x in rows)/1e6)
