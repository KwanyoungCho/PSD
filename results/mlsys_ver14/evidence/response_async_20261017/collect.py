"""Copy completed immutable runs only; keep in-progress files off analysis."""
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
SSH=['ssh','-S','/tmp/duet-gap-eslab18-20261017.sock','-o','BatchMode=yes','-p','1100',
     'chokwans99@115.145.135.238']
REMOTE='/data2/chokwans99/response-async-20261017'


def copy(source,destination,recursive=False):
    args=['scp','-q','-o','ControlPath=/tmp/duet-gap-eslab18-20261017.sock','-P','1100']
    if recursive:args.append('-r')
    subprocess.run(args+['chokwans99@115.145.135.238:'+source,str(destination)],check=True)


def main():
    code="""from pathlib import Path
import json
r=Path('/data2/chokwans99/response-async-20261017')
rows=[]
for p in sorted((r/'runs').iterdir()):
 s=p/'status.json'
 rows.append(dict(tag=p.name,status=json.loads(s.read_text()) if s.exists() else dict(status='running')))
print(json.dumps(rows))
"""
    rows=json.loads(subprocess.check_output(SSH+['python3','-'],input=code,text=True))
    target=HERE/'raw/runs';target.mkdir(parents=True,exist_ok=True)
    for item in rows:
        if item['status']['status']!='complete':continue
        tag=item['tag'];assert all(c.isalnum() or c=='_' for c in tag)
        local=target/tag
        if not (local/'status.json').exists():copy(REMOTE+'/runs/'+tag,target,True)
    (HERE/'STATUS.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Complete:',sum(x['status']['status']=='complete' for x in rows),'Running:',
          [x['tag'] for x in rows if x['status']['status']=='running'])


if __name__=='__main__':main()
