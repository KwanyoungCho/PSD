"""Archive measured evidence only; model weights are never included."""
import hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def archive(name,paths):
 dest=HERE/name;members={}
 with tarfile.open(dest,'w:gz',compresslevel=6) as tar:
  for path,root in paths:
   items=sorted(path.rglob('*')) if path.is_dir() else [path]
   for p in items:
    if not p.is_file():continue
    arc=str(Path(root)/p.relative_to(path)) if path.is_dir() else root
    members[arc]=dict(bytes=p.stat().st_size,sha256=sha(p));tar.add(p,arcname=arc,recursive=False)
 return dict(archive=dest.name,sha256=sha(dest),bytes=dest.stat().st_size,members=members)

def main():
 result=json.loads((HERE/'RESULTS.json').read_text());assert result['status']=='complete'
 base=Path('/data/chokwans99');entries=[]
 folders=[('RAW_L2.tar.gz','ver02-local-algorithms-full-l2-final-20261014'),
          ('RAW_L3.tar.gz','ver02-local-algorithms-full-l3-20261014'),
          ('RAW_L3_RECHECK.tar.gz','ver02-local-algorithms-l3-recheck3-20261014'),
          ('RAW_SMOKES.tar.gz','ver02-local-algorithms-smoke-20261014'),
          ('RAW_FINAL_GREEDY.tar.gz','ver02-local-algorithms-final-greedy-20261014'),
          ('RAW_PROFILES.tar.gz','ver02-local-algorithms-profiles-20261014')]
 for name,folder in folders:
  path=base/folder
  assert path.is_dir(),path
  jobs=json.loads((path/'campaign.json').read_text())
  if name=='RAW_L3.tar.gz':
   # Original contaminated attempts are intentionally retained; clean retries
   # are selected by RESULTS.json and archived separately.
   assert all((j['status']=='complete' and j['returncode']==0) or j['external_pids'] for j in jobs),folder
  else:
   assert all(j['status']=='complete' and j['returncode']==0 for j in jobs),folder
  entries.append(archive(name,[(path,folder)]))
 checks=[(HERE/n,n) for n in ('checks_frozen','checks','checks_initial')]
 entries.append(archive('CHECK_LOGS.tar.gz',checks))
 entries.append(archive('CONTROL_LOGS.tar.gz',[(p,p.name) for p in sorted(HERE.glob('*.log'))]))
 attempts=[]
 for name in ('ver02-local-algorithms-full-l2-20261014','ver02-local-algorithms-full-l2-retry-20261014','ver02-local-algorithms-full-l2-clean-20261014','ver02-local-algorithms-l3-recheck-20261014','ver02-local-algorithms-l3-recheck2-20261014'):
  path=base/name
  for p in path.iterdir():
   # Baseline was copied byte-for-byte to the final folder; do not archive it four times.
   if p.name.startswith('root_only') or p.name in ('plan.json','campaign.json'):
    attempts.append((p,str(Path(name)/p.name)))
 entries.append(archive('EXCLUDED_ATTEMPTS.tar.gz',attempts))
 (HERE/'ARCHIVES.json').write_text(json.dumps(entries,indent=2)+'\n')
 for n in ('CHECKS.json','CHECKS_PROVENANCE.json'):
  (HERE/n).write_bytes((HERE/'checks_frozen'/n).read_bytes())
 print([(x['archive'],round(x['bytes']/2**20,2)) for x in entries])

if __name__=='__main__':main()
