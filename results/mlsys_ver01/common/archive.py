"""Archive this common-fix campaign without changing historical evidence."""
import hashlib,json,subprocess,tarfile,tempfile
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2]
folders=('abba','smoke','tree_r0','tree_r1','temperature','chain','features','prefix','bulk','profiles','post_log_smoke')
files=sorted({p for f in folders for p in (R/f).rglob('*') if p.is_file()} | {p for p in R.glob('*.log') if p.name!='archive.log'})
rows=[dict(path=str(p.relative_to(R)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
archive=R/'RAW_RESULTS.tar.gz'
with tarfile.open(archive,'w:gz') as out:
    for p in files:out.add(p,arcname=str(p.relative_to(R)))
with tarfile.open(archive) as inp:
    assert len(inp.getmembers())==len(rows)
    for row in rows:assert hashlib.sha256(inp.extractfile(row['path']).read()).hexdigest()==row['sha256']
(R/'RAW_MANIFEST.json').write_text(json.dumps(dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=rows,read_back_verified=True),indent=2))
for revision,name in [('a82f7d2','RUNTIME.patch'),('3d51664','COMMON_ONLY.patch')]:
    diff=subprocess.check_output(['git','diff',revision,'--','ssd/ssd','ssd/bench/bench.py'],cwd=ROOT)
    (R/name).write_bytes(diff)
    # Check each patch applies to the advertised clean source revision.
    paths=subprocess.check_output(['git','diff','--name-only',revision,'--','ssd/ssd','ssd/bench/bench.py'],cwd=ROOT,text=True).splitlines()
    with tempfile.TemporaryDirectory(prefix='duet_common_patch_') as td:
        dest=Path(td)
        for p in paths:
            existing=subprocess.run(['git','show',revision+':'+p],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            if existing.returncode==0:
                f=dest/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(existing.stdout)
        subprocess.run(['git','apply','--unsafe-paths',str((R/name).resolve())],cwd=dest,check=True)
        for p in paths:assert (dest/p).read_bytes()==(ROOT/p).read_bytes(),p
    print('PATCH VERIFIED',revision,len(paths))
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'ssd/ssd').rglob('*.py')}
(R/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,prompt_sha256=hashlib.sha256((R.parent/'questions.json').read_bytes()).hexdigest()),indent=2))
(R/'PATCH_CHECK.json').write_text(json.dumps(dict(common_base='3d51664',paper_base='a82f7d2',both_patches_apply_and_match=True),indent=2))
print('ARCHIVE VERIFIED',len(rows),archive.stat().st_size)
