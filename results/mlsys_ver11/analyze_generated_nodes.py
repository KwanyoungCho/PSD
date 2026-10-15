"""Read-only reanalysis of v0.2 equal-budget forest traces (G nonbinding)."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import gzip, json, hashlib
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/mlsys_ver02/equal_budget_20261014/RESULTS.json'

def scan(job):
 host,name,path=job
 hist={}; total=0
 with gzip.open(path,'rt') as f:
  for line in f:
   x=json.loads(line);total+=1
   assert x['generated']==x['processed']*x['C']
   assert x['served']==min(x['generated'],64 if name.endswith('64') else 32)
   for rank in ('all',str(x['root'])):
    key=f"p{x['phase']}_rank_{rank}"
    hist.setdefault(key,Counter())[x['generated']]+=1
 def stats(h):
  n=sum(h.values());pairs=sorted(h.items())
  def q(p):
   rank=max(1,__import__('math').ceil(n*p));acc=0
   for val,count in pairs:
    acc+=count
    if acc>=rank:return val
  return dict(n=n,mean=sum(k*v for k,v in h.items())/n,p50=q(.5),p95=q(.95),maximum=max(h),histogram=dict(pairs))
 with Path(path).open('rb') as raw:
  digest=hashlib.file_digest(raw,'sha256').hexdigest()
 return dict(host=host,name=name,path=str(path),sha256=digest,records=total,stats={k:stats(v) for k,v in hist.items()})

if __name__=='__main__':
 data=json.loads(SOURCE.read_text());jobs=[]
 for host,h in data.items():
  for name,a in h['arms'].items():
   if name.startswith('chain'):continue
   path=Path(a['raw_path']).parent/(name+'_profile')/'equal_forest.jsonl.gz'
   jobs.append((host,name,path))
 with ProcessPoolExecutor(max_workers=4) as pool:
  rows=list(pool.map(scan,jobs))
 result=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  scope='All exported root trees, including warmup and final work; root token excluded. Not conditioned on actual hit.',runs=rows)
 (Path(__file__).parent/'GENERATED_NODES.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in rows:
  print(r['host'],r['name'],[(p,r['stats'][f'p{p}_rank_0']['mean'],r['stats'][f'p{p}_rank_all']['maximum']) for p in (1,2)],flush=True)
