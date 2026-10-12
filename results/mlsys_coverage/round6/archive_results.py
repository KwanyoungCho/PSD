"""Lossless archive of immutable completed/failed runs; preserve raw local files."""
import argparse,gzip,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');a=p.parse_args()
archive=HERE/'archives';archive.mkdir(exist_ok=True)
manifest=HERE/'ARCHIVE_MANIFEST.json'
if a.restore:
    entries=json.loads(manifest.read_text())
    for e in entries:
        dst=HERE/e['path'];src=archive/(e['path']+'.gz')
        raw=gzip.decompress(src.read_bytes())
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError(src)
        if dst.exists() and dst.read_bytes()!=raw:raise ValueError(f'Existing raw file differs: {dst}')
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
        os.utime(dst,ns=(e['mtime_ns'],e['mtime_ns']))
    print('Restored',len(entries));raise SystemExit
paths=[]
for path in HERE.glob('*/*.json'):
    if path.parent.name=='archives' or path.name in ('plan.json','campaign.json'):continue
    d=json.loads(path.read_text())
    manifest_path=path.parent/'campaign.json'
    job=next((r for r in json.loads(manifest_path.read_text()) if r['name']==path.stem),{}) if manifest_path.exists() else {}
    finished=d.get('status') in ('complete','failed') if isinstance(d,dict) else False
    finished=finished or job.get('status') in ('failed','timeout','incomplete')
    if isinstance(d,dict) and finished and 'cells' in d:paths.append(path)
paths+=list((HERE/'profiles').rglob('*.json'))
entries=[]
for path in sorted(paths):
    raw=path.read_bytes();rel=str(path.relative_to(HERE));dst=archive/(rel+'.gz')
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
    if gzip.decompress(dst.read_bytes())!=raw:raise ValueError(f'Archive mismatch: {path}')
    entries.append(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
        compressed_bytes=dst.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
manifest.write_text(json.dumps(entries,indent=2))
(HERE/'.gitignore').write_text('# Raw files preserved losslessly in archives; run archive_results.py --restore.\n__pycache__/\n'+''.join('/'+e['path']+'\n' for e in entries))
print('Archived',len(entries),'raw MiB',round(sum(e['bytes'] for e in entries)/2**20,1),'compressed MiB',round(sum(e['compressed_bytes'] for e in entries)/2**20,1))
