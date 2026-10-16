"""Copy only immutable completed runs; leave growing run files on the server."""
import argparse,json,subprocess
from pathlib import Path
R='/data2/chokwans99/target-gap-20261017'
SSH=['ssh','-S','/tmp/duet-gap-eslab18-20261017.sock','-p','1100','chokwans99@115.145.135.238']
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');args=p.parse_args()
code='import pathlib,json; r=pathlib.Path('+repr(R)+'); print(json.dumps([f.parent.name for f in (r/"runs").glob("*/status.json") if json.loads(f.read_text())["status"]=="complete"]))'
names=json.loads(subprocess.check_output(SSH+['python3 -c '+"'"+code.replace("'","'\\''")+"'"],text=True))
raw=HERE/'raw';(raw/'runs').mkdir(parents=True,exist_ok=True)
pending=[n for n in names if args.all or not (raw/'runs'/n/'status.json').exists()]
if pending:
 subprocess.run(['scp','-q','-r','-o','ControlPath=/tmp/duet-gap-eslab18-20261017.sock','-P','1100',*[f'chokwans99@115.145.135.238:{R}/runs/{n}' for n in pending],str(raw/'runs')],check=True)
print(json.dumps(dict(complete=len(names),copied=pending)),flush=True)
