import json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
while not (R/'MICRO.json').exists():time.sleep(5)
time.sleep(3)
subprocess.run([sys.executable,str(R.parent/'run_pair.py'),'--plan',str(R/'temperature_plan.json'),'--gpus','6,7','--port','33900','--directory',str(R/'temperature')],cwd=R.parents[2],check=True)
