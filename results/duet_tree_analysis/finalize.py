"""Check campaign completeness and record reproducibility fingerprints."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
folders=['probe','rollout_q_path','rollout_phase_sibling_q_bin']
counts={};snapshot_count=0
for name in folders:
    folder=HERE/name
    if json.loads((folder/'completion.json').read_text())['exit_code']!=0:raise ValueError(name)
    x=json.loads((folder/'manifest.json').read_text())
    if len(x['records'])!=len(x['plan']['mapping']):raise ValueError('Incomplete '+name)
    if any(r['output_tokens']!=128 for r in x['records']):raise ValueError('Output count')
    for meta in x['snapshots']:
        path=folder/meta['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Checksum '+str(path))
    counts[name]=dict(prompts=len(x['records']),snapshots=len(x['snapshots']))
    snapshot_count+=len(x['snapshots'])
for seed in [921,922]:
    for policy in ['q_path','phase_sibling_q_bin']:
        folder=HERE/f'bench_{seed}_{policy}'
        if json.loads((folder/'completion.json').read_text())['exit_code']!=0:raise ValueError(str(folder))
        x=json.loads((folder/'result.json').read_text())
        if len(x['records'])!=16 or any(r['output_tokens']!=128 for r in x['records']):raise ValueError('Benchmark count')
        counts[folder.name]=dict(prompts=16,output_tokens=2048)
sources=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[p for p in HERE.glob('*.json') if p.name!='source_manifest.json']
sources += [ROOT/p for p in ['ssd/ssd/engine/helpers/p2_tree.py','ssd/ssd/engine/helpers/p2_tree_executor.py',
    'ssd/ssd/engine/helpers/p1_tree.py','ssd/ssd/engine/helpers/tree_rerank_gpu.py','ssd/ssd/engine/draft_runner.py',
    'ssd/ssd/engine/verifier.py','ssd/ssd/config.py']]
sources += [p for folder in [HERE/name for name in folders]+list(HERE.glob('bench_*')) if folder.is_dir()
            for p in folder.glob('*.json')]
fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))}
out=dict(date='2026-09-21',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         note='Current source fingerprints; pre-existing dirty production source preserved. Runtime score hook is opt-in.',
         campaigns=counts,verified_snapshot_checksums=snapshot_count,sha256=fingerprints)
(HERE/'source_manifest.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(campaigns=counts,verified_snapshot_checksums=snapshot_count,fingerprinted_files=len(fingerprints)),indent=2))
