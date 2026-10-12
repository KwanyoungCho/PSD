"""Post-hoc shape/context diagnosis; never replaces the full-population result."""
import collections,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
a=json.loads((R/'ANALYSIS.json').read_text())
rows={x['name']:x for x in a['rows']};results=[]

def bins(row):
 d=json.loads((R/row['file']).read_text());out=collections.defaultdict(list)
 if row['batch']!=1:return out
 for group in d['groups']:
  length=len(d['prompt_ids'][group['index']])
  for s in group['steps']:
   if s['batch']==1 and not s['cap'] and not s['clipped'] and s['events'] and s['verify']:
    e=s['events'][0];key=(int(e['source']),int(e['valid_k']),length//128)
    out[key].append(s['verify'][0]*1000)
   length+=s['emitted']
 return out

for pair in a['pairs']:
 if pair['scope']!='full' or pair['batch']!=1:continue
 base=bins(rows[pair['base']]);candidate=bins(rows[pair['candidate']])
 common=set(base)&set(candidate);counts={k:(len(base[k])+len(candidate[k]))/2 for k in common}
 total=sum(counts.values());table=[]
 for k in sorted(common):
  x,y=statistics.mean(base[k]),statistics.mean(candidate[k])
  table.append(dict(source=k[0],valid_k=k[1],prefix_bin128=k[2],base_n=len(base[k]),candidate_n=len(candidate[k]),
                    base_ms=x,candidate_ms=y,change_ms=y-x,pooled_weight=counts[k]/total))
 b=sum(x['base_ms']*x['pooled_weight'] for x in table);c=sum(x['candidate_ms']*x['pooled_weight'] for x in table)
 by_source=[]
 for source in (0,1,2):
  xs=[v for k,ls in base.items() if k[0]==source for v in ls]
  ys=[v for k,ls in candidate.items() if k[0]==source for v in ls]
  if xs and ys:by_source.append(dict(source=source,base_n=len(xs),candidate_n=len(ys),
      base_ms=statistics.mean(xs),candidate_ms=statistics.mean(ys),change_ms=statistics.mean(ys)-statistics.mean(xs)))
 results.append(dict(contrast=pair['contrast'],rep=pair['rep'],base=pair['base'],candidate=pair['candidate'],
    overlap_base_fraction=sum(len(base[k]) for k in common)/sum(map(len,base.values())),
    overlap_candidate_fraction=sum(len(candidate[k]) for k in common)/sum(map(len,candidate.values())),
    standardized_base_ms=b,standardized_candidate_ms=c,standardized_change_pct=100*(c/b-1),
    phase_means=by_source,bins=table))
(R/'STRATIFIED.json').write_text(json.dumps(dict(scope='Post-hoc matching of phase, valid proposal width, and 128-token prefix bins. Shared-support diagnostic only; primary estimates include all eligible steps.',pairs=results),indent=2))
print('Stratified pairs:',len(results))
