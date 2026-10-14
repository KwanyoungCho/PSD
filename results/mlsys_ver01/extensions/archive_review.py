"""Preserve final source and raw evidence without modifying git history."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-root',type=Path,required=True)
    args=p.parse_args()
    here=Path(__file__).resolve().parent
    root=here.parents[2]
    expected=json.loads((here/'FINAL_RUNTIME_HASHES.json').read_text())
    assert all(digest(root/name)==value for name,value in expected.items()), 'Runtime changed since validation'
    manifest=dict(base='f12d0f1da6a2e4a26e9cb11ddcca54c189736d19',
        source_port='1d9ace6ccfb774132ef0149426436dee676f8a03',runtime_frozen=True,
        state='working-tree changes; not committed or pushed by this packaging step',
        raw_root=str(args.raw_root),archives={},raw_files={},source_files={})
    folders=['nova_latest','llama2_full_v2','llama2_sensitivity','accept_components',
             'target_components','b1_regression']
    rawfiles=[q for name in folders for q in (args.raw_root/name).rglob('*')
              if q.is_file() and '__pycache__' not in q.parts]
    rawfiles.append(args.raw_root/'component_llama3_4090_v2.json')
    with tarfile.open(here/'RAW_RESULTS.tar.gz','w:gz',compresslevel=6) as tf:
        for f in sorted(rawfiles):
            name=str(f.relative_to(args.raw_root));tf.add(f,arcname=name)
            manifest['raw_files'][name]=dict(bytes=f.stat().st_size,sha256=digest(f))
    sourcefiles=list((root/'ssd/ssd').rglob('*.py'))+list((root/'ssd/tests').glob('*.py'))
    sourcefiles+=list((root/'ssd/bench').glob('mlsys*.py'))
    sourcefiles+=[root/'results/mlsys_ver01/run_pair.py',root/'results/mlsys_ver01/questions.json']
    with tarfile.open(here/'FINAL_SOURCE.tar.gz','w:gz',compresslevel=6) as tf:
        for f in sorted(set(sourcefiles)):
            name=str(f.relative_to(root));tf.add(f,arcname=name)
            manifest['source_files'][name]=dict(bytes=f.stat().st_size,sha256=digest(f))
    patch=subprocess.check_output(['git','diff','--binary',manifest['base'],'--','ssd/ssd'],cwd=root)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','ssd/ssd'],cwd=root,text=True).splitlines()
    for name in untracked:
        diff=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],cwd=root,stdout=subprocess.PIPE)
        assert diff.returncode in (0,1)
        patch+=diff.stdout
    (here/'RUNTIME.patch').write_bytes(patch)
    for name in ('RAW_RESULTS.tar.gz','FINAL_SOURCE.tar.gz','RUNTIME.patch','FINAL_ANALYSIS.json','figs/REVIEW.pdf'):
        path=here/name;manifest['archives'][name]=dict(bytes=path.stat().st_size,sha256=digest(path))
    (here/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['archives'],indent=2))


if __name__=='__main__':main()
