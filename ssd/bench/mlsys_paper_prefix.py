"""Full-model prefix validation on the paper engine with only KV repairs."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ.get('PAPER_CHECKOUT','/home/chokwans99/PSD-paper-review')+'/ssd')
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
os.environ['PAPER_COLD_PREFIX']='0'
from mlsys_paper_hooks import install
install('kv')
import torch
from ssd import LLM,SamplingParams

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--prompts',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 result=dict(status='running',model=a.model,cases=[],scope='Original paper engine plus four KV repairs; cached vs correct cold outputs')
 def save():a.output.write_text(json.dumps(result,indent=2))
 save();llm=None
 try:
  llm=LLM(a.model,num_gpus=1,max_num_seqs=8,max_model_len=1024,max_num_batched_tokens=8192,
          gpu_memory_utilization=.45,speculate=False)
  source=json.loads(a.prompts.read_text());texts=[x if isinstance(x,str) else x['turns'][0] for x in source[:8]]
  encoded=[llm.tokenizer.encode(x) for x in texts];params=SamplingParams(temperature=0,max_new_tokens=8,ignore_eos=True)
  allocation=[];allocate=llm.scheduler.block_manager.allocate
  def observe(s):
   allocate(s);allocation.append(dict(cached=s.num_cached_tokens,blocks=list(s.block_table)))
  llm.scheduler.block_manager.allocate=observe
  for b in (1,8):
   llm.scheduler.max_num_seqs=b
   for length in (512,768,800):
    prompts=[(x*((length+len(x)-1)//len(x)))[:length] for x in encoded[:b]]
    llm.generate(prompts,params,use_tqdm=False)
    for rep in range(3):
     arms={}
     for name in ('cold','cached') if rep%2==0 else ('cached','cold'):
      if name=='cold':llm.scheduler.block_manager.hash_to_block_id.clear()
      else:llm.generate(prompts,params,use_tqdm=False)
      allocation.clear();torch.cuda.synchronize();t=time.perf_counter()
      out,metrics=llm.generate(prompts,params,use_tqdm=False);torch.cuda.synchronize()
      arms[name]=dict(outputs=[x['token_ids'] for x in out],prefill_ms=metrics['prefill_total_time']*1000,
                     wall_ms=(time.perf_counter()-t)*1000,allocation=list(allocation))
     same=arms['cold']['outputs']==arms['cached']['outputs']
     result['cases'].append(dict(batch=b,length=length,repeat=rep,arms=arms,outputs_equal=same));save()
     if not same:raise AssertionError('KV reuse changes greedy output')
  result['status']='complete';save()
 except BaseException as exc:result.update(status='failed',error=repr(exc));save();raise
 finally:
  if llm:llm.exit(hard=False)

if __name__=='__main__':main()
