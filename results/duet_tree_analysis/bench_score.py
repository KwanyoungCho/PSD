"""Uninstrumented per-prompt benchmark; only opt-in expansion score changes."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'ssd'));sys.path.insert(0,str(ROOT/'ssd/bench'))
if os.environ.get('DUET_TREE_SCORE_MODE','q_path')!='q_path':
    from score_hook import install
    install()


def main():
    import bench
    plan=json.loads((HERE/'benchmark_plan.json').read_text())
    original=bench.run_benchmark
    def run(args,llm,prompts,sampling_params):
        original(args,llm,prompts[:1],sampling_params[:1])
        records=[];outputs=[]
        if len(prompts)!=len(plan['mapping']):raise ValueError('Prompt count')
        for i,(prompt,param) in enumerate(zip(prompts,sampling_params)):
            if hashlib.sha256(json.dumps(prompt).encode()).hexdigest()!=plan['mapping'][i]['token_sha256']:
                raise ValueError('Prompt hash mismatch')
            bench.reset_metrics()
            out,elapsed,metrics=original(args,llm,[prompt],[param])
            emitted=sum(len(x['token_ids']) for x in out)
            if emitted!=128:raise ValueError('Output count')
            records.append(dict(prompt=i,output_tokens=emitted,wall_s=elapsed,metrics=json.loads(json.dumps(metrics))))
            outputs.extend(out)
            payload=dict(schema='untraced_tree_score_ablation_v1',args=vars(args),
                         policy=os.environ['DUET_TREE_SCORE_MODE'],plan=plan,records=records)
            dst=Path(os.environ['DUET_TREE_BENCH_OUT']);tmp=dst.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload,indent=2));tmp.replace(dst)
        return outputs,sum(r['wall_s'] for r in records),records[-1]['metrics']
    bench.run_benchmark=run;bench.main()


if __name__=='__main__':main()
