"""Prepare the reviewed local campaign only after a clean published release."""
from pathlib import Path
import gzip,hashlib,json,os,shutil,socket,subprocess


def main():
    code=Path(os.environ['VER15_CHECKOUT']).resolve()
    raw=Path(os.environ['VER15_RAW']).resolve();report=Path(os.environ['VER15_REPORT']).resolve()
    if socket.gethostname().split('.')[0]!='orion':raise RuntimeError('Review local GPU/CPU layout on another server')
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/MLSys-ver1.5'],cwd=code,text=True).split()[0]
    assert commit==remote,'Push the reviewed release before starting the campaign'
    base=code/'results/mlsys_ver15';checks=json.loads((base/'checks/CHECKS.json').read_text())
    assert checks['status']=='pass'
    for path,digest in checks['runtime_sha256'].items():
        assert hashlib.sha256((code/path).read_bytes()).hexdigest()==digest,path
    raw.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    if (raw/'QUEUE.json').exists():raise RuntimeError('Existing campaign: inspect status/STOP; do not overwrite')
    tools=base/'tools'
    assert (tools/'adapter/ssd/ssd/utils/performance.py').read_bytes()==(code/'ssd/ssd/utils/performance.py').read_bytes()
    for name in ('benchmark_wall.py','capacity_adapter.py','run_once_helper.py'):
        shutil.copyfile(tools/name,raw/name)
    if (raw/'adapter').exists():
        # Preserve old smoke provenance, but serve only the reviewed minimum
        # dependency set, so unused benchmark modules cannot be imported.
        import time
        archive=raw/'development_checks'/f'adapter_before_release_{time.time_ns()}'
        archive.parent.mkdir(exist_ok=True)
        shutil.move(str(raw/'adapter'),str(archive))
    shutil.copytree(tools/'adapter',raw/'adapter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    with gzip.open(tools/'INPUTS_FULL.json.gz','rb') as stream:(raw/'INPUTS_FULL.json').write_bytes(stream.read())
    assert hashlib.sha256((raw/'INPUTS_FULL.json').read_bytes()).hexdigest()==checks['input_sha256']
    for name in ('PROTOCOL.json','DATASET.json'):shutil.copyfile(base/'checks'/name,raw/name)
    template=dict(checkout=str(code),mode='tree',profile=False,async_send=1,replica=0,proxy_stream=0,
        config=dict(exit_layer=21,k1=10,k2=5,p1_verify=23,p2_verify=15,roots=4,siblings=3,
        ds_width=56,ps_width=28,ps_roots=24,passes=1,output_tokens=1024,input_cap=0,warm_prompts=3,
        prompts='INPUTS_FULL.json',limit=0,seed=42,temperature=.7))
    def dump(name,value):(raw/name).write_text(json.dumps(value,indent=2)+'\n')
    dump('PLAN.json',dict(gpu_pairs=[[4,5],[6,7],[1,2],[0,3]],job_template=template))
    dump('ADAPTER_SHA256.json',{str(f.relative_to(raw)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in raw.rglob('*.py') if 'runs' not in f.parts})
    dump('RELEASE.json',dict(commit=commit,validation_passed=True,pushed=True,
        checks_sha256=hashlib.sha256((base/'checks/CHECKS.json').read_bytes()).hexdigest(),
        tree_selection='path_score_top_m',verification='target_sample_match'))
    print(json.dumps(dict(prepared=str(raw),report=str(report),commit=commit)))


if __name__=='__main__':main()
