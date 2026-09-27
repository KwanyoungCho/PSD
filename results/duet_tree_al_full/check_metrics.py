"""Hand-calculated checks for AL units, final-step censoring and zero-hit questions."""
from analyze import question_counts,aggregate

def event(source,accepted):
    return dict(source=source,accepted_spec_len=accepted,accepted_len=accepted+1,
                cache_hit=int(source!=0),valid_k={0:2,1:8,2:6}[source])

questions=[dict(question_id=1,group='mt_bench'),dict(question_id=2,group='qa')]
def row(q,events):
    length=sum(e['accepted_len'] for e in events)-int(events[-1]['accepted_len']>1)
    return dict(question_id=q,output_tokens=length,wall_s=1.,stop='eos',input_tokens=20,
                metrics=dict(phase_events=events))
rows=[row(1,[event(0,1),event(1,2),event(2,2)]),
      row(1,[event(2,1),event(1,4)]),row(2,[event(0,0),event(1,0)])]
counts=question_counts(rows,questions);out=aggregate(list(counts.values()))
assert counts[1]['turns']==2 and counts[2]['hit_n']==0
assert out['al_hit']==1.5 and out['al_p1']==2 and out['al_p2']==1
assert out['al_all_step_sensitivity']==1.8
assert out['question_seed_zero_hits']==1
assert out['histogram']==[0,1,1,0,0]
assert out['raw_hit_n']==5 and out['hit_n']==2
assert out['mean_hit_verify_nodes']==7
assert out['phase_histograms']['1']==[0,0,1,0,0]
assert out['phase_tails']['2']['1']==1 and out['phase_tails']['2']['2']==0
assert out['terminal_events']==3
assert out['al_emitted_all_steps']==1.4
print('AL units, two-turn pooling, terminal-step exclusion and undefined-zero-hit handling: PASS')
