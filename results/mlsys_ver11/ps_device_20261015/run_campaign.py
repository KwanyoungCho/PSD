"""Sequential paired runs; telemetry is observational and both arms identical."""
import json,subprocess,threading,time
from pathlib import Path
RAW=Path('/data/chokwans99/duet-ps-device-20261015')
stop=threading.Event()
def telemetry():
    with (RAW/'telemetry.jsonl').open('w') as f:
        while not stop.is_set():
            row={'time':time.time(), 'monotonic_ns':time.monotonic_ns()}
            for k,query in [('gpus','--query-gpu=index,uuid,name,clocks.sm,temperature.gpu,utilization.gpu,memory.used'),('processes','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory')]:
                r=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],capture_output=True,text=True)
                row[k]=r.stdout;row[k+'_returncode']=r.returncode
            f.write(json.dumps(row)+'\n');f.flush();stop.wait(2)
t=threading.Thread(target=telemetry);t.start()
try:
    for name in ('run_profiles.sh','run_unprofiled.sh'):
        subprocess.run(['bash',str(RAW/name)],check=True)
finally:
    stop.set();t.join()
