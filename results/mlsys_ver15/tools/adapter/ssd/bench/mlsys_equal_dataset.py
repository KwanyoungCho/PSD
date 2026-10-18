"""Input identities for paper-format flattened Spec-Bench benchmark records.

Historical paper inputs treat each supplied MT turn as a separate raw prompt.
No response history is synthesized. A zero input cap means no truncation.
"""
import json
import hashlib
from pathlib import Path


def load(path, tokenizer, limit=0, input_cap=0):
    source=json.loads(Path(path).read_text())
    if input_cap < 0:raise ValueError('Input cap must be nonnegative')
    indices=list(range(len(source)))
    if 0 < limit < len(source):indices=[i*len(source)//limit for i in range(limit)]
    structured=bool(source) and isinstance(source[0],dict) and 'prompt' in source[0]
    metadata=[];prompts=[]
    vocab_hash=None;vocab_size=None
    for i in indices:
        item=source[i]
        if structured:
            for key in ('uid','question_id','group','turn','prompt','question_index'):
                if key not in item:raise ValueError('Incomplete paper input identity: '+key)
            text=item['prompt']
            metadata.append({k:item[k] for k in ('uid','question_id','group','turn','question_index')})
        else:
            text=item if isinstance(item,str) else item['turns'][0]
        if structured and 'input_ids' in item:
            if vocab_hash is None:
                vocab=tokenizer.get_vocab();vocab_size=max(vocab.values())+1
                vocab_hash=hashlib.sha256(json.dumps(vocab,sort_keys=True).encode()).hexdigest()
            if item.get('vocab_sha256')!=vocab_hash:
                raise ValueError('Paper input token IDs use a different vocabulary')
            tokens=item['input_ids']
            if not tokens or any(type(t) is not int or t < 0 or t >= vocab_size for t in tokens):
                raise ValueError('Invalid pretokenized paper input')
        else:
            tokens=tokenizer.encode(text)
        prompts.append(tokens[:input_cap] if input_cap else tokens)
    if structured and len({r['uid'] for r in metadata})!=len(metadata):
        raise ValueError('Duplicate input UID')
    return prompts,indices,metadata


def attach(records, metadata, phase):
    if not metadata:return
    if len(records)!=len(metadata):raise ValueError('Request identity count mismatch')
    for r,m in zip(records,metadata):
        r.update(uid=phase+':'+m['uid'],paper_uid=m['uid'],question_id=str(m['question_id']),
                 question_index=m['question_index'],turn_index=m['turn'],group=m['group'])
