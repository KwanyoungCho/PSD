"""Lossless Git-friendly benchmark/profile JSON archive; raw files stay local.

--restore reconstitutes the paths consumed by the analysis scripts.
Full-vocabulary NPZ calibration snapshots remain local; compact sufficient
statistics, frozen tables and source hashes are versioned separately.
"""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');a=p.parse_args()
folder=HERE/'archives';folder.mkdir(exist_ok=True)
if a.restore:
 expected={x['path']:x['sha256'] for x in json.loads((HERE/'ARCHIVE_MANIFEST.json').read_text())}
 for path in folder.rglob('*.gz'):
  dest=HERE/path.relative_to(folder).with_suffix('')
  key=str(dest.relative_to(HERE))
  if dest.exists():
   if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected[key]:
    raise ValueError(f'Existing raw file differs from archived manifest: {dest}')
   continue
  dest.parent.mkdir(parents=True,exist_ok=True)
  with gzip.open(path,'rb') as src,dest.open('wb') as out:shutil.copyfileobj(src,out)
  if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected[key]:
   raise ValueError(f'Restored file checksum mismatch: {dest}')
else:
 manifest=[]
 for path in sorted(HERE.glob('*/*.json')):
  if path.parent.name in ['calibration','archives','dense_calibration']:continue
  if path.name in ['plan.json','campaign.json']:continue
  try:d=json.loads(path.read_text())
  except (json.JSONDecodeError,UnicodeDecodeError):continue
  if not isinstance(d,dict) or d.get('status')!='complete' or 'cells' not in d:continue
  rel=path.relative_to(HERE);dest=folder/(str(rel)+'.gz');dest.parent.mkdir(parents=True,exist_ok=True)
  raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
  if not dest.exists():dest.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
  if hashlib.sha256(gzip.decompress(dest.read_bytes())).hexdigest()!=digest:
   raise ValueError(f'Existing archive differs: {dest}')
  manifest.append(dict(path=str(rel),sha256=digest,bytes=len(raw),compressed_bytes=dest.stat().st_size))
 for path in sorted(list((HERE/'profiles').rglob('duet_profile*.json'))+
                    list((HERE/'kernels').rglob('*.trace.json'))):
  rel=path.relative_to(HERE);dest=folder/(str(rel)+'.gz');dest.parent.mkdir(parents=True,exist_ok=True)
  raw=path.read_bytes()
  if not dest.exists():dest.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
  if gzip.decompress(dest.read_bytes())!=raw:raise ValueError(f'Archive mismatch {dest}')
  manifest.append(dict(path=str(rel),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),compressed_bytes=dest.stat().st_size))
 (HERE/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 print('Archived files',len(manifest),'raw MiB',round(sum(x['bytes'] for x in manifest)/2**20,1),'gzip MiB',round(sum(x['compressed_bytes'] for x in manifest)/2**20,1))
