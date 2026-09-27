"""Freeze full-corpus, AL-first comparison before running either policy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-new-tokens',type=int,default=128)
    args=ap.parse_args()
    if (HERE/'plan.json').exists():raise ValueError('Plan already frozen')
    source=Path('/home/chokwans99/dev/llm/specinfer.cpp/examples/spec-bench-data/question.jsonl')
    raw=source.read_bytes();questions=[json.loads(s) for s in raw.decode().splitlines() if s.strip()]
    url='https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl'
    official=urllib.request.urlopen(url,timeout=30).read()
    online=[json.loads(s) for s in official.decode().splitlines() if s.strip()]
    fields=lambda rows:[{k:r[k] for k in ['question_id','category','turns']} for r in rows]
    if fields(questions)!=fields(online):raise ValueError('Local questions differ from upstream')
    groups=['mt_bench','translation','summarization','qa','math_reasoning','rag']
    for i,q in enumerate(questions):
        q['index']=i;q['group']=q['category'] if q['category'] in groups[1:] else 'mt_bench'
    if len(questions)!=480 or Counter(q['group'] for q in questions)!=Counter({g:80 for g in groups}):
        raise ValueError('Expected 480 questions, six groups of 80')
    turns=sum(len(q['turns']) for q in questions)
    if turns!=560:raise ValueError('Expected 560 turns')
    (HERE/'questions.json').write_text(json.dumps(questions,ensure_ascii=False,indent=2))
    (HERE/'upstream_question.jsonl').write_bytes(official)
    from transformers import AutoTokenizer
    target='/home/chokwans99/awq_calibrated/layerskip_llama2_70b'
    tok=AutoTokenizer.from_pretrained(target,local_files_only=True)
    lens=[len(tok.encode(s,add_special_tokens=False)) for q in questions for s in q['turns']]
    # Later turns append actual previous generated token IDs and raw new question.
    max_bound=max(sum(len(tok.encode(s,add_special_tokens=False)) for s in q['turns'])+
                  len(q['turns'])*args.max_new_tokens+16*len(q['turns'])+128 for q in questions)
    context=2048 if max_bound<=2048 else 4096
    if max_bound>4096:raise ValueError('Full context cannot fit without truncation; review required')
    previous=ROOT/'results/duet_tree_analysis'
    calibration=previous/'calibration_frozen.json'
    fitting_texts=[json.loads(s)['text'] for s in (previous/'datasets/alpaca/alpaca_data_10000.jsonl').read_text().splitlines()][:8]
    overlap=[q['question_id'] for q in questions if any(s.strip() in {t.strip() for t in fitting_texts} for s in q['turns'])]
    if overlap:raise ValueError('Exact calibration-prompt overlap: '+str(overlap))
    policies=['q_path','phase_sibling_q_bin'];seeds=[1,42,123]
    jobs=[dict(seed=seed,policy=p) for i,seed in enumerate(seeds) for p in (policies if i%2==0 else policies[::-1])]
    plan=dict(schema='duet_full_al_v1',date='2026-09-21',objective='Maximize accepted tree descendants conditional on cache hit at fixed root policy and fixed generation/verification budgets. TPS is descriptive only.',
        source=dict(local=str(source),official_url=url,local_sha256=hashlib.sha256(raw).hexdigest(),
                    downloaded_sha256=hashlib.sha256(official).hexdigest(),question_content_equal=True),
        questions=480,turns=560,groups=groups,group_counts=dict(Counter(q['group'] for q in questions)),
        questions_sha256=hashlib.sha256((HERE/'questions.json').read_bytes()).hexdigest(),
        policies=policies,seeds=seeds,jobs=jobs,max_new_tokens=args.max_new_tokens,ignore_eos=False,
        max_model_len=context,extend_draft_rope=context>2048,prompt_input_token_range=[min(lens),max(lens)],
        context_upper_bound=max_bound,truncate_inputs=False,multiturn='Raw prompt tokens on turn 0; prior input plus actual generated tokens plus newline and next question on subsequent turns. Both turns clustered by original question ID.',
        calibration_path=str(calibration),calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        calibration_exact_overlap_question_ids=overlap,calibration_refit=False,
        config=dict(exit_layer=56,k1=4,k2=2,p1_fanout=3,p2_budget=15,root_count=15,roots_per_position=3,
                    c_tensor=3,p1_generation_nodes=8,p1_verify_nodes=8,p2_generation_nodes=6,p2_verify_nodes=6,
                    allocation='dynamic',root_source='unchanged existing legacy proxy residual',temperature=.7,batch=1),
        metrics=dict(primary='Equal-six-task mean of per-task hit-conditional descendant AL, pooled over prespecified seeds; exclude final verification step of each request to avoid EOS/cap censoring.',
            phase_metrics=['P1 conditional AL','P2 conditional AL','equal-phase AL as mix-control diagnostic'],
            additional=['all-step AL sensitivity','hit rate','accepted-length histogram and tail probabilities','question-level macro AL on jointly observed questions','generation/verification node counts','TPS descriptive only'],
            inference='Paired stratified bootstrap of original questions within each task, keeping both turns and all seeds together; 5000 resamples. Also report seed-specific and task-specific differences.',
            no_hit='Zero-hit questions remain in full coverage counts and ratio statistics; undefined per-question conditional AL is missing, never zero-imputed.',
            thresholds='No policy retuning or early efficacy stopping on this corpus.'),
        runtime=dict(gpus=[3,4,5,6,7],draft_kv_memory_fraction=.5 if context==2048 else .30,
                     workspace_mb=128,warmup_pages='all',profiling=False,full_distribution_observer=False),
        limits=['Fixed depth 4/2 and node budgets 8/6; not global AL-optimal architecture search.',
                'Root policy is fixed, but realized contexts/cache-hit mix may differ across trajectories.',
                'Previously used research benchmark, not globally unseen data; frozen scalar calibration uses separate prompts.'])
    (HERE/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    # A single placeholder lets bench initialize the normal engine; run_full loads all questions itself.
    data=HERE/'dataset/alpaca/alpaca_data_10000.jsonl';data.parent.mkdir(parents=True,exist_ok=True)
    data.write_text(json.dumps(dict(text=questions[0]['turns'][0]),ensure_ascii=False)+'\n')
    print(json.dumps({k:plan[k] for k in ['questions','turns','group_counts','max_new_tokens','max_model_len','context_upper_bound','calibration_exact_overlap_question_ids']},indent=2))


if __name__=='__main__':main()
