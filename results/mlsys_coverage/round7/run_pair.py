"""Sequential original-paper A/B jobs on an otherwise idle GPU pair."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def query(*args):return subprocess.check_output(['nvidia-smi',*args],text=True).strip()

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--gpus',required=True)
 p.add_argument('--port',type=int,required=True);p.add_argument('--directory',type=Path,required=True)
 a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
 root=Path(__file__).resolve().parents[3];bench=root/'ssd/bench/mlsys_paper_ab.py'
 jobs=json.loads(a.plan.read_text());(a.directory/'plan.json').write_text(json.dumps(jobs,indent=2))
 uuids=set(query('-i',a.gpus,'--query-gpu=uuid','--format=csv,noheader').splitlines())
 def procs():return [r for r in query('--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits').splitlines() if r.split(',')[0] in uuids]
 manifest_path=a.directory/'campaign.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
 for i,job in enumerate(jobs):
  dest=a.directory/(job['name']+'.json');log=a.directory/(job['name']+'.log')
  if dest.exists():
   if json.loads(dest.read_text()).get('status')=='complete':
    print('SKIP_COMPLETE',job['name'],flush=True);continue
   raise FileExistsError(dest)
  if procs():raise RuntimeError('GPU pair busy: '+repr(procs()))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,SSD_DIST_PORT=str(a.port+i),SSD_SHM_NAME=f'paper7_{a.port+i}',
      SSD_CUDA_ARCH='8.9',SSD_ATTN_BACKEND='auto',OMP_NUM_THREADS='4',SSD_SEED=str(job.get('draft_seed',91)),
      SSD_TREE_EXEC='1',SSD_CHAIN_PROXY_GRAPH='1',SSD_TREE_PROXY_GRAPH='1',SSD_PROFILE_DUET='0',
      SSD_ASYNC_PROXY_SEND='1',SSD_PROXY_STREAM='0',PAPER_ARM=job['arm'],PAPER_COLD_PREFIX='1')
  env.update(job.get('env',{}))
  selected_bench=bench.with_name(job.get('bench',bench.name))
  cmd=[sys.executable,'-O',str(selected_bench),*job['args'],'--output',str(dest.resolve())]
  row=dict(name=job['name'],command=cmd,arm=job['arm'],start=time.time(),external_pids=[],gpu_pair=a.gpus)
  print('START',job['name'],flush=True)
  with log.open('w') as out,(a.directory/(job['name']+'.telemetry.jsonl')).open('w') as tr:
   proc=subprocess.Popen(cmd,env=env,cwd=root,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while proc.poll() is None:
     active=procs();foreign=[]
     for line in active:
      pid=int(line.split(',')[1])
      try:
       if os.getpgid(pid)!=proc.pid:foreign.append(pid)
      except ProcessLookupError:pass
     row['external_pids']=sorted(set(row['external_pids']+foreign))
     tr.write(json.dumps(dict(time=time.time(),processes=active,foreign=foreign,
         gpu=query('-i',a.gpus,'--query-gpu=index,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used','--format=csv,noheader')))+'\n');tr.flush()
     if time.time()-row['start']>2400:
      row['timeout']=True;break
     time.sleep(5)
   finally:
    # Controller I/O failures must not leave its GPU workers spinning.
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=3)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
   proc.wait();row.update(exit_code=proc.returncode,end=time.time())
  data=json.loads(dest.read_text()) if dest.exists() else {}
  row['status']=data.get('status','no_output');manifest.append(row)
  manifest_path.write_text(json.dumps(manifest,indent=2))
  print('END',job['name'],row['status'],proc.returncode,flush=True)
  if proc.returncode or row['status']!='complete':raise RuntimeError('Job failed; preserve and diagnose '+job['name'])
  time.sleep(2)

if __name__=='__main__':main()
