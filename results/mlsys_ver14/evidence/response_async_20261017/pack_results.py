"""Package raw evidence and verify every archived regular file byte-for-byte."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent


def main():
    raw=HERE/'raw';archive=HERE/'RAW_RESULTS.tar.xz'
    files=sorted(p for p in raw.rglob('*') if p.is_file())
    expected={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with tarfile.open(archive,'w:xz',preset=3) as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(HERE)))
    actual={}
    with tarfile.open(archive,'r:xz') as tar:
        for member in tar:
            if member.isfile():actual[member.name]=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
    assert actual==expected,'Archived evidence differs'
    result=dict(status='pass',files=len(files),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                archive_bytes=archive.stat().st_size,completed_model_runs=32,failed_parity_runs=['hostlist_13b_host_a'],
                raw_sha256=expected)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Archive verified:',len(files),'files,',archive.stat().st_size,'bytes',flush=True)


if __name__=='__main__':main()
