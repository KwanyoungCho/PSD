"""CPU audit of actual MT inputs, outputs, warmup and context-limit accounting."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ssd/bench'))
from specbench_history import BaseTokenHistory, PROTOCOL, load_turns


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(report, rows, tokenizer):
    require(report['status']=='complete', 'Run is not complete')
    require(report['experiment_protocol']['input_protocol']['protocol']==PROTOCOL, 'Wrong input protocol')
    args=report['args']
    history=BaseTokenHistory(rows,tokenizer,context=report['kwargs']['max_model_len'],output_cap=args['output_tokens'])
    ledger=report['turn_ledger']
    require(len(ledger)==len(rows)==len(report['prompt_ids']), 'Incomplete turn population')
    groups=report['warm_groups']+report['groups']
    by_index={g['index']:g for g in groups}
    require(len(by_index)==len(groups), 'Duplicate generated turn')
    reasons=Counter(); second=0; first=0; skipped=[]
    for i,(source,actual) in enumerate(zip(rows,ledger)):
        prepared=history.prepare(i)
        for key,value in prepared.items():
            require(actual.get(key)==value, f'{source["uid"]}: wrong {key}')
        require(report['prompt_ids'][i]==prepared['input_ids'], 'Reported model input differs from history')
        phase='warm' if i<args['warm_prompts'] else 'measured_0'
        require(actual['phase']==phase, 'Wrong warmup population')
        completed=history.complete(actual['output_ids'],
            'context_limit_guard' if actual['finish_reason']=='context_limit_guard' else None)
        for key,value in completed.items():
            require(actual.get(key)==value, f'{source["uid"]}: wrong completed {key}')
        reasons[actual['finish_reason']]+=1
        if actual['admission_reason']:
            skipped.append(actual['uid'])
            require(i not in by_index, 'Unadmitted turn has fabricated model timing')
            require(actual['performance_excluded'] is True, 'Unadmitted turn must be excluded')
        else:
            require(i in by_index, 'Generated turn is missing its measurement')
            group=by_index[i]
            require(group['phase']==phase and group['outputs']==[actual['output_ids']], 'Output/phase mismatch')
            request=group['request_records'][0]
            require(group['n']==1 and len(group['request_records'])==1, 'Expected B1')
            require(request['paper_uid']==actual['uid'], 'Wrong request identity')
            require(request['input_tokens']==len(prepared['input_ids']), 'Engine input length mismatch')
            require(request['end_context_length']==len(prepared['input_ids'])+len(actual['output_ids']), 'Engine context accounting mismatch')
            require(request['output_ids']==actual['output_ids'], 'Request/generated answer mismatch')
            require(actual['performance_excluded']==(phase=='warm'), 'Wrong warmup exclusion')
        if source['turn']:
            second+=1
        else:
            first+=1
    require(report['outputs']==[g['outputs'][0] for g in report['groups']], 'Measured output accounting mismatch')
    require(report['conversation_population']['unadmitted_uids']==skipped, 'Missing context exclusions')
    return dict(state='passed',protocol=PROTOCOL,turns=len(rows),first_turns=first,
        followup_turns=second,generated_turns=len(groups),warmup_generated=len(report['warm_groups']),
        measured_generated=len(report['groups']),finish_reasons=dict(reasons),unadmitted_uids=skipped,
        exact_answer_history=True,other_task_frozen_inputs_preserved=True,
        note='Input/accounting validation, not a performance comparison')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--target',help='Tokenizer location override when moving between servers')
    parser.add_argument('--prompts',type=Path,help='Dataset location override')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with (gzip.open(args.run,'rt') if args.run.suffix=='.gz' else args.run.open()) as f:
        report=json.load(f)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.target or report['args']['target'],use_fast=True)
    source=args.prompts or Path(report['args']['prompts'])
    if not source.is_absolute() and not source.exists():
        source=ROOT/source
    rows=load_turns(source,report['args']['limit'],report['args'].get('task','all'))
    result=validate(report,rows,tokenizer)
    if args.output:
        with args.output.open('x') as f:
            json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
