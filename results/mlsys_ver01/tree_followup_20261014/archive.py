"""Archive source provenance and raw evidence without model weights."""
from pathlib import Path
import tarfile,hashlib,json,difflib,subprocess
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2];RAW=Path('/data/chokwans99/duet-tree-20261014')
roots={'baseline':Path('/data/chokwans99/PSD-tree-frozen-20261014'),'candidate':Path('/data/chokwans99/PSD-tree-candidate-20261014'),'stream':Path('/data/chokwans99/PSD-tree-stream-20261014'),'final':REPO}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={tag:{str(f.relative_to(root)):digest(f) for f in (root/'ssd/ssd').rglob('*.py')} for tag,root in roots.items()}
patch=[];changes=[]
for rel in sorted(files['final']):
 if files['baseline'].get(rel)==files['final'][rel]:continue
 a=roots['baseline']/rel;b=REPO/rel;changes.append(rel)
 patch.extend(difflib.unified_diff(a.read_text().splitlines(True) if a.exists() else [],b.read_text().splitlines(True),fromfile='a/'+rel if a.exists() else '/dev/null',tofile='b/'+rel))
(HERE/'ADOPTED_RUNTIME.patch').write_text(''.join(patch))
subprocess.run(['git','apply','--check',str(HERE/'ADOPTED_RUNTIME.patch')],cwd=roots['baseline'],check=True)
with tarfile.open(HERE/'SOURCE_SNAPSHOTS.tar.gz','w:gz',compresslevel=6) as tar:
 for tag,root in roots.items():
  for rel in sorted(files[tag]):tar.add(root/rel,arcname=tag+'/'+rel,recursive=False)
 for folder in ['ssd/bench','ssd/tests']:
  for f in (REPO/folder).glob('*.py'):tar.add(f,arcname='review/'+str(f.relative_to(REPO)),recursive=False)
 for f in HERE.glob('*.py'):tar.add(f,arcname='analysis/'+f.name,recursive=False)
 tar.add(REPO/'results/mlsys_ver01/questions.json',arcname='review/questions.json')
raw_hash={}
with tarfile.open(HERE/'RAW_RESULTS.tar.gz','w:gz',compresslevel=6) as tar:
 for f in sorted(RAW.rglob('*')):
  if not f.is_file() or f.suffix=='.tmp':continue
  rel=str(f.relative_to(RAW));raw_hash[rel]=digest(f);tar.add(f,arcname=rel,recursive=False)
verified={};status={}
for f in RAW.rglob('campaign.json'):
 rows=json.loads(f.read_text());status[str(f.relative_to(RAW))]=rows
for f in RAW.rglob('*.json'):
 try:j=json.loads(f.read_text())
 except Exception:continue
 if not isinstance(j,dict) or j.get('status')!='complete' or 'source_hashes' not in j:continue
 exact=[tag for tag,hashes in files.items() if j['source_hashes']==hashes]
 verified[str(f.relative_to(RAW))]=exact
manifest={'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'committed':False,'adopted_runtime_files':changes,'source_hashes':files,'raw_sha256':raw_hash,'runtime_snapshot_matches':verified,'campaigns':status,'artifacts':{name:dict(bytes=(HERE/name).stat().st_size,sha256=digest(HERE/name)) for name in ['SOURCE_SNAPSHOTS.tar.gz','RAW_RESULTS.tar.gz','ADOPTED_RUNTIME.patch']},'notes':['Before is f12d0f1 plus previously reviewed uncommitted extensions, not the original B1 paper engine.','Final differs from the stream snapshot only in the first-round warmup page bound. The evaluated 2048 configurations keep the same page families; final unaligned-cap 2040 generation also passed.','Per-run adapter hashes are in raw JSON. Exploratory adapters evolved; final source snapshots and final integration runs are the primary reproduction evidence.','Cold-prefix benchmark population and cap/clip/partial/preparation exclusions are described in REPORT.md.']}
(HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('adopted files',len(changes),'runtime source matches',sum(bool(x) for x in verified.values()),'/',len(verified))
print('archives',manifest['artifacts'])
