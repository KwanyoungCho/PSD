"""Verify measured runtime hashes against the commits each run names."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2]
checks=[];cache={}
for p in sorted(R.glob('*/*.json')):
    if p.name in ('plan.json','campaign.json'):continue
    try:d=json.loads(p.read_text())
    except (ValueError,UnicodeError):continue
    if not isinstance(d,dict) or d.get('status')!='complete' or 'source_hashes' not in d:continue
    if p.parent.name=='smoke':continue # implementation smoke explicitly predates runtime commit
    rev=d['paper_commit']
    if rev not in cache:
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',rev,'--','ssd/ssd'],cwd=ROOT,text=True).splitlines()
        cache[rev]={name:hashlib.sha256(subprocess.check_output(['git','show',rev+':'+name],cwd=ROOT)).hexdigest() for name in names if name.endswith('.py')}
    diff=[name for name,h in d['source_hashes'].items() if cache[rev].get(name)!=h]
    assert not diff,(str(p),rev,diff)
    checks.append(dict(run=str(p.relative_to(R)),revision=rev,runtime_matches_commit=True,files=len(d['source_hashes'])))
(R/'SOURCE_AUDIT.json').write_text(json.dumps(checks,indent=2));print('Source commits verified for',len(checks),'completed runs')
