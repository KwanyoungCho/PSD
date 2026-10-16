"""Check elapsed entire LLMEngine.step calls independently of native counters."""
import json
from pathlib import Path
from analyze_steps import load,index_steps,assert_parity
P=Path(__file__).resolve().parent;rows=[]
for model,variant,stage in [('7b','restore','final'),('13b','restore','final'),('7b_chain','inputs','perf')]:
 counts=0;before=after=0
 for repeat in ['a','b']:
  a=load(P/'raw',f'before_{model}_{stage}_{repeat}');b=load(P/'raw',f'{variant}_{model}_{stage}_{repeat}');assert_parity(a,b)
  aa=index_steps(a);bb=index_steps(b)
  for k in aa.keys()&bb.keys():
   before+=(aa[k]['audit_end_ns']-aa[k]['audit_start_ns'])/1e6
   after+=(bb[k]['audit_end_ns']-bb[k]['audit_start_ns'])/1e6
   counts+=1
 rows.append(dict(model=model,variant=variant,matched_steps=counts,before_ms=before/counts,after_ms=after/counts,reduction_pct=100*(1-after/before)))
(P/'WALL_STEP_CHECK.json').write_text(json.dumps(dict(comparisons=rows,note='External monotonic wrapper covers the entire original LLMEngine.step call, including bookkeeping outside its internal timer. Same 2x-selected paired cohort.'),indent=2)+'\n')
print(rows)
