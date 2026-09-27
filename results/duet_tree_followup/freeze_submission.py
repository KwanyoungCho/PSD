"""Freeze executable inputs before starting an unattended GPU wait queue."""
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    dest=HERE/'execution_manifest.json'
    if dest.exists():raise FileExistsError('Submission is already frozen')
    plan=json.loads((HERE/'plan.json').read_text())
    checks=['math_checks.json','frontier_checks.json','cuda_checks.json','analysis_checks.json']
    checks += [f'executor_{p}.json' for p in plan['policies']]
    for name in checks:
        if not json.loads((HERE/name).read_text())['passed']:raise ValueError('Required gate failed '+name)
    files=list(HERE.glob('*.py'))+[HERE/name for name in checks]
    files += [HERE/name for name in ('plan.json','questions.json','gain_calibration.json')]
    files += list((ROOT/'ssd/ssd').rglob('*.py'))+list((ROOT/'ssd/bench').glob('*.py'))
    for directory in ('duet_tree_analysis','duet_tree_al_full','duet_tree_posthoc'):
        files += list((HERE.parent/directory).glob('*.py'))
    files.append(Path(plan['calibration_path']))
    weights={}
    for folder in ('/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0',
                   '/home/chokwans99/awq_calibrated/layerskip_llama2_70b',
                   '/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4'):
        for path in Path(folder).rglob('*'):
            if path.is_file() and path.suffix in ('.json','.safetensors','.pt','.bin'):
                stat=path.stat();weights[str(path)]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    result=dict(frozen_unix=time.time(),files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},
        external_weight_metadata=weights,
        scope='Execution source/calibration hashes and external model artifact size/mtime; not a result-completion manifest.')
    dest.write_text(json.dumps(result,indent=2));print('Frozen',len(result['files']),'files and',len(weights),'model artifacts')


if __name__=='__main__':main()
