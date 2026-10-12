"""Repeated-prompt prefill diagnosis, separate from speculative tree KV promotion."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd import LLM,SamplingParams

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True)
    p.add_argument('--prompts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=dict(status='running',model=a.model,gpu=torch.cuda.get_device_name(),cases=[],
        scope='Repeated-prompt AR prefill; cold cache lookup cleared, warm lookup retained. Both use current correct attention. No algorithm rollback.')
    def save():a.output.write_text(json.dumps(result,indent=2))
    save();llm=None
    try:
        llm=LLM(a.model,num_gpus=1,max_num_seqs=8,max_model_len=1024,max_num_batched_tokens=8192,
                gpu_memory_utilization=.45,speculate=False,enforce_eager=False)
        source=json.loads(a.prompts.read_text());tok=llm.tokenizer
        texts=[s if isinstance(s,str) else s['turns'][0] for s in source]
        encoded=[tok.encode(t) for t in texts[:8]]
        params=SamplingParams(temperature=0,max_new_tokens=8,ignore_eos=True)
        alloc=llm.scheduler.block_manager.allocate;observations=[]
        def observe(s):
            alloc(s);observations.append(dict(cached=s.num_cached_tokens,total=len(s),blocks=list(s.block_table)))
        llm.scheduler.block_manager.allocate=observe
        for b in (1,8):
          llm.scheduler.max_num_seqs=b
          for length in (512,768,800):
            prompts=[(ids*((length+len(ids)-1)//len(ids)))[:length] for ids in encoded[:b]]
            # Shapes and kernel compilation warm before timing.
            llm.generate(prompts,params,use_tqdm=False)
            for rep in range(3):
              results={}
              order=('cold','cached') if rep%2==0 else ('cached','cold')
              for mode in order:
                if mode=='cold':llm.scheduler.block_manager.hash_to_block_id.clear()
                else:llm.generate(prompts,params,use_tqdm=False)
                observations.clear();torch.cuda.synchronize();start=time.perf_counter()
                outputs,metrics=llm.generate(prompts,params,use_tqdm=False)
                torch.cuda.synchronize()
                results[mode]=dict(prefill_ms=metrics['prefill_total_time']*1000,wall_ms=(time.perf_counter()-start)*1000,
                    allocation=list(observations),outputs=[o['token_ids'] for o in outputs],preemptions=metrics['preemptions'])
              result['cases'].append(dict(batch=b,length=length,repeat=rep,arms=results,
                  greedy_tokens_equal=results['cold']['outputs']==results['cached']['outputs']))
              save();print('PREFIX_CASE',b,length,rep,{k:v['prefill_ms'] for k,v in results.items()},result['cases'][-1]['greedy_tokens_equal'],flush=True)
        result['status']='complete';save()
    except BaseException as e:result.update(status='failed',error=f'{type(e).__name__}: {e}');save();raise
    finally:
        if llm:llm.exit(hard=False)

if __name__=='__main__':main()
