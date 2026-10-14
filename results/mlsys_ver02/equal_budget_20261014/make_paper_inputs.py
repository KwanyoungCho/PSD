"""Recover the historical paper's exact raw text and tokenizer convention."""
import json,collections,hashlib
from pathlib import Path
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OLD=Path('/home/chokwans99/PSD')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((ROOT/'results/mlsys_ver01/questions.json').read_text())
modeltok=AutoTokenizer.from_pretrained('/home/chokwans99/models_mlsys/layerskip-llama2-7b')
papertok=AutoTokenizer.from_pretrained('/home/chokwans99/awq_calibrated/layerskip_llama2_70b')
assert modeltok.get_vocab()==papertok.get_vocab()
vocab_hash=hashlib.sha256(json.dumps(modeltok.get_vocab(),sort_keys=True).encode()).hexdigest()
rows=[dict(uid=str(r['question_id'])+'_t'+str(i),question_id=r['question_id'],group=r['group'],category=r['category'],turn=i,prompt=t,question_index=k,input_ids=papertok.encode(t),vocab_sha256=vocab_hash) for k,r in enumerate(q) for i,t in enumerate(r['turns'])]
oldpath=OLD/'ssd/experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_full_specbench_seed42_20260811/p1_backbone_s42_o1024.jsonl';old=[json.loads(x) for x in oldpath.open()]
assert len(old)==len(rows)==560
assert all(all(r[k]==o[k] for k in ('uid','prompt','group','question_id','turn')) for r,o in zip(rows,old))
assert all(len(r['input_ids'])==o['prefill_total_tokens'] for r,o in zip(rows,old))
length=[len(r['input_ids']) for r in rows]
report=dict(pdf=str(next(OLD.glob('(KCC*.pdf'))),pdf_conditions=dict(temperature=.7,max_new_tokens=1024,batch=1,input_truncation='not specified'),paper_script='a82f7d2:ssd/experiments/proxy_async_overlap/tree_sweep/tree_param_postmortem_20260814/run_full_arm.sh',script_conditions=dict(template='raw',max_model_len=2048,top_p=1.,max_new_tokens=1024,data_turns=560),raw_input_match=dict(source=str(oldpath),source_sha256=sha(oldpath),turns=len(old),uid_prompt_group_mismatches=0,input_token_length_mismatches=0),dataset=dict(questions=480,turns=560,group_turn_counts=dict(collections.Counter(r['group'] for r in rows)),max_input_tokens=max(length),inputs_over_512=sum(x>512 for x in length)),tokenizer=dict(vocab_equal=True,vocab_sha256=vocab_hash,default_7b_tokenization_different_turns=sum(modeltok.encode(r['prompt'])!=r['input_ids'] for r in rows),policy='freeze paper-tokenizer input IDs; same vocabulary/ID mapping as selected 7B model'),mt_note='Historical input is flattened: each MT turn is an independent raw prompt, without generated conversation history. Match this protocol; do not describe it as conversation-history evaluation.')
(HERE/'SPEC_BENCH_FULL.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
report['dataset']['sha256']=sha(HERE/'SPEC_BENCH_FULL.json')
(HERE/'PAPER_PROTOCOL.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
