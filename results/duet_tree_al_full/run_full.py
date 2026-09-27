"""All questions/turns, natural EOS, fixed policy; streaming complete records."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'ssd'));sys.path.insert(0,str(ROOT/'ssd/bench'))
from runtime import install
install()  # Also executes in spawned workers, before engine construction.


def main():
    import bench
    from transformers import AutoTokenizer
    from ssd.sampling_params import SamplingParams
    plan=json.loads((HERE/'plan.json').read_text())
    questions=json.loads((HERE/'questions.json').read_text())
    if hashlib.sha256((HERE/'questions.json').read_bytes()).hexdigest()!=plan['questions_sha256']:raise ValueError('Question checksum')
    dest=Path(os.environ['DUET_FULL_OUT']);dest.mkdir(parents=True,exist_ok=True)
    records_path=dest/'records.jsonl'
    if records_path.exists():raise ValueError('Refuse to append a second trajectory to existing records')
    smoke=int(os.environ.get('DUET_FULL_SMOKE','0'))
    if smoke:
        chosen=[questions[0]]+[next(q for q in questions if q['group']==g) for g in plan['groups'][1:]]
        chosen.append(max(questions,key=lambda q:sum(len(s) for s in q['turns'])))
        questions=list({q['question_id']:q for q in chosen}.values())
    tok=AutoTokenizer.from_pretrained('/home/chokwans99/awq_calibrated/layerskip_llama2_70b',local_files_only=True)
    original=bench.run_benchmark
    def run(args,llm,prompts,params):
        warm=SamplingParams(temperature=.7,max_new_tokens=min(64,plan['max_new_tokens']),ignore_eos=False)
        original(args,llm,prompts[:1],[warm])
        count=0;outputs=[];sum_wall=0.;start=time.time()
        with records_path.open('x',buffering=1) as f:
            for q in questions:
                history=[]
                for turn,text in enumerate(q['turns']):
                    current=tok.encode(text,add_special_tokens=False)
                    prompt=current if turn==0 else history+tok.encode('\n\n',add_special_tokens=False)+current
                    if len(prompt)+plan['max_new_tokens']+128>plan['max_model_len']:
                        raise ValueError(f'Context overflow without truncation: {q["question_id"]}, turn {turn}')
                    param=SamplingParams(temperature=.7,max_new_tokens=plan['max_new_tokens'],ignore_eos=False)
                    bench.reset_metrics()
                    out,elapsed,metrics=original(args,llm,[prompt],[param])
                    ids=out[0]['token_ids'];metrics=json.loads(json.dumps(metrics))
                    if not 0<len(ids)<=plan['max_new_tokens']:raise ValueError('Output length')
                    if len(ids)<plan['max_new_tokens'] and ids[-1]!=tok.eos_token_id:
                        raise ValueError('Unexpected early stop (neither EOS nor cap)')
                    events=metrics['phase_events']
                    # Immediate EOS during prefill is a valid zero-hit request.
                    if not events and not (len(ids)==1 and ids[-1]==tok.eos_token_id):
                        raise ValueError('No verification events without immediate EOS')
                    for e in events:
                        a=e['accepted_spec_len']
                        if a<0:raise ValueError('Negative acceptance')
                        if e['cache_hit'] and a>({1:4,2:2}[e['source']]):raise ValueError('AL exceeds fixed depth')
                    record=dict(uid=f'{q["question_id"]}_t{turn}',question_id=q['question_id'],group=q['group'],category=q['category'],turn=turn,
                        seed=args.seed,policy=os.environ['DUET_TREE_SCORE_MODE'],input_tokens=len(prompt),
                        input_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),output_tokens=len(ids),
                        output_ids=ids,output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        stop='eos' if ids[-1]==tok.eos_token_id else 'cap',wall_s=elapsed,metrics=metrics)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                    history=prompt+ids;count+=1;sum_wall+=elapsed;outputs=out
                    status=dict(completed_questions=q['index']+1,completed_turns=count,total_turns=sum(len(x['turns']) for x in questions),
                        elapsed_s=time.time()-start,last_uid=record['uid'],smoke=bool(smoke))
                    tmp=dest/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(dest/'status.json')
                    print('[full AL]',os.environ['DUET_TREE_SCORE_MODE'],args.seed,count,status['total_turns'],record['uid'],len(ids),flush=True)
        expected=[f'{q["question_id"]}_t{t}' for q in questions for t in range(len(q['turns']))]
        actual=[json.loads(s)['uid'] for s in records_path.read_text().splitlines()]
        if actual!=expected:raise ValueError('Incomplete or reordered corpus')
        (dest/'validated.json').write_text(json.dumps(dict(turns=count,questions=len(questions),
            records_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest(),smoke=bool(smoke)),indent=2))
        # bench.main only knows the single initialization placeholder. Its
        # generic throughput/generation display would mislabel this corpus.
        print('[full AL complete]',len(questions),'questions',count,'turns',flush=True)
        raise SystemExit(0)
    bench.run_benchmark=run
    bench.main()


if __name__=='__main__':main()
