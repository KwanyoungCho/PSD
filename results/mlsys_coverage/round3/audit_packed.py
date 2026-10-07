"""Align existing 48-prompt AR data with full packed-tree greedy outputs.

The extracted files are correctness-only subsets; they contain no performance
metrics. HF independently checks each first divergence, using the AR prefix.
"""
import hashlib,json,subprocess,sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]

def wait_for_gpu():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible or ',' in visible:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one free GPU for the HF audit')
    query=lambda args:subprocess.check_output(['nvidia-smi',*args],text=True).strip()
    uuids=set(query(['-i',visible,'--query-gpu=uuid','--format=csv,noheader']).splitlines())
    start=time.monotonic()
    while True:
        rows=query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']).splitlines()
        busy=[r for r in rows if r.split(',')[0] in uuids]
        if not busy:return
        if time.monotonic()-start>1800:raise TimeoutError(f'Audit GPU remains busy: {busy}')
        time.sleep(5)

for name in ['llama2','llama3']:
    raw_path=ROOT/'packed_comparison'/f'{name}_tree_packed.json'
    raw=json.loads(raw_path.read_text())
    if raw['status']!='complete':raise ValueError(f'Incomplete run: {raw_path}')
    ar_path=ROOT/f'{name}_ar'/f'{name}_ar_reference.json'
    ar=json.loads(ar_path.read_text())
    positions={tuple(p):i for i,p in enumerate(raw['prompt_ids'])}
    select=[positions[tuple(p)] for p in ar['prompt_ids']]
    cells=[c for c in raw['cells'] if c['temperature']==0]
    subset=dict(audit_subset_only=True,source=str(raw_path.relative_to(REPO)),
        selected_raw_indexes=select,args=raw['args'],prompt_ids=[raw['prompt_ids'][i] for i in select],
        prompt_sha256=ar['prompt_sha256'],cells=[dict(batch=cells[0]['batch'],temperature=0,
            seed=cells[0]['seed'],outputs=[cells[0]['outputs'][i] for i in select])])
    assert hashlib.sha256(json.dumps(subset['prompt_ids']).encode()).hexdigest()==subset['prompt_sha256']
    dst=ROOT/f'{name}_packed_greedy_subset.json';dst.write_text(json.dumps(subset,indent=2))
    out=ROOT/f'{name}_packed_greedy_audit.json'
    wait_for_gpu()
    with (ROOT/f'{name}_packed_greedy_audit.log').open('w') as log:
        subprocess.run([sys.executable,str(REPO/'ssd/bench/mlsys_greedy_audit.py'),'--ar',str(ar_path),
            '--duet',str(dst),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    audit=json.loads(out.read_text());records=audit['divergences']
    result=dict(model=name,checked=len(select),exact=len(select)-len(records),
        full_greedy_repeat_exact=sum(a['token_ids']==b['token_ids'] for a,b in zip(cells[0]['outputs'],cells[1]['outputs'])),
        full_greedy_repeat_total=len(cells[0]['outputs']),
        max_selected_logit_gap=max([max(r['hf_top_logits'])-r['duet_logit'] for r in records if 'duet_logit' in r] or [0]))
    # Also inspect EVERY first divergence from the prior full tree run,
    # rather than extrapolating the 48-prompt AR audit to all 480 prompts.
    full=dict(audit_subset_only=True,source=str(raw_path.relative_to(REPO)),
        args=raw['args'],prompt_ids=raw['prompt_ids'],prompt_sha256=raw['prompt_sha256'],
        cells=[dict(batch=cells[0]['batch'],temperature=0,seed=cells[0]['seed'],outputs=cells[0]['outputs'])])
    full_path=ROOT/f'{name}_packed_greedy_full_audit_input.json'
    full_path.write_text(json.dumps(full,indent=2))
    tree_path=ROOT/f'{name}_full1'/f'{name}_tree_full.json'
    full_out=ROOT/f'{name}_packed_vs_tree_full_audit.json'
    wait_for_gpu()
    with (ROOT/f'{name}_packed_vs_tree_full_audit.log').open('w') as log:
        subprocess.run([sys.executable,str(REPO/'ssd/bench/mlsys_greedy_audit.py'),'--ar',str(tree_path),
            '--duet',str(full_path),'--reference-batch','8','--output',str(full_out)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    full_records=json.loads(full_out.read_text())['divergences']
    result.update(full_tree_reference_checked=len(full['prompt_ids']),
        full_tree_reference_exact=len(full['prompt_ids'])-len(full_records),
        full_tree_reference_max_selected_gap=max([max(r['hf_top_logits'])-r['duet_logit'] for r in full_records if 'duet_logit' in r] or [0]),
        full_tree_reference_all_top5=all(r.get('duet_token') in r.get('hf_top_ids',[]) for r in full_records))
    print(result,flush=True)
    (ROOT/f'{name}_packed_greedy_summary.json').write_text(json.dumps(result,indent=2))
