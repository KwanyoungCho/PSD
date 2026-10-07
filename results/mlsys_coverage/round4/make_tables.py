"""Readable tables and prespecified matched-node contrasts from saved reports."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'RESULTS.json').read_text()); ix={(r['group'],r['name']):r for r in rows}
lines=['# Round4 집계표','', '각 표는 완료된 실행만 포함한다. AL은 recovery를 포함하며 `AL*`는 cap/clip terminal event를 제외한 값이다. TPS는 실제 반환 token 기준 전체 decode TPS다. B8의 후속 pass는 같은 engine의 2·3회차 평균이며 독립 process 반복이 아니다. CI는 질문 단위 paired bootstrap 2,000회이며 여러 정책 비교에 대한 보정은 하지 않았다.','']
def add(s=''):lines.append(s)
def ci(x):return '—' if x is None else f'[{x[0]:+.4f}, {x[1]:+.4f}]'
def val(x,fmt='.3f'):return '—' if x is None else format(x,fmt)
def table(head):add('| '+' | '.join(head)+' |');add('|'+'|'.join(['---']*len(head))+'|')
add('## 1. B8 tree 정책: 전체 480 질문 × target seed 3개');add()
table(['모델','정책','AL*','q-path 대비','ΔAL* 95% CI','원시 AL','후속 pass TPS'])
for m in ['llama2','llama3']:
 for n in ['q_path','reach','q_gain','reach_gain','reach_frontier','reach_gain_frontier','dense_reach','dense_reach_gain_frontier']:
  r=ix.get((m+'_full',n))
  if r:add('| '+' | '.join([m,n,val(r['boundary_excluded_al']),val(r.get('clean_al_relative_pct'),'+.2f')+'%',ci(r.get('delta_clean_al_ci95')),val(r['al']),val(r['later_pass_tps_mean'],'.1f')])+' |')
add();add('Dense 보정은 사전 지정 8개 질문으로 고정했다. 아래는 두 arm 모두에서 보정 질문을 제거한 472개다.');add()
table(['모델','정책','AL*','baseline AL*','상대 차이','ΔAL* CI'])
for m in ['llama2','llama3']:
 for n in ['dense_reach','dense_reach_gain_frontier']:
  r=ix.get((m+'_full',n))
  if r and 'heldout472' in r:
   h=r['heldout472'];add(f'| {m} | {n} | {h["boundary_excluded_al"]:.3f} | {h["reference_boundary_excluded_al"]:.3f} | {h["clean_relative_pct"]:+.2f}% | {ci(h["clean_delta_ci95"])} |')
add();add('## 2. B1 legacy 경로에서 이전 tree 정책 적용');add()
table(['모델','정책','AL*','상대 차이','ΔAL* CI','TPS (1회)'])
for m in ['llama2','llama3']:
 for n in ['legacy_q_path','legacy_reach_gain_frontier']:
  r=ix.get((m+'_b1',n))
  if r:add(f'| {m} | {n} | {r["boundary_excluded_al"]:.3f} | {r.get("clean_al_relative_pct",0):+.2f}% | {ci(r.get("delta_clean_al_ci95"))} | {r["tps_mean"]:.1f} |')
add();add('## 3. Miss fallback: 같은 unified 경로에서 비교');add()
table(['모델','B','miss 정책','AL*','chain2 대비 ΔAL* CI','miss 조건부 원시 AL','P1/P2 hit','TPS'])
contrasts=[]
for m in ['llama2','llama3']:
 for b in [1,8]:
  group=m+('_b1' if b==1 else '_full')
  for n,label in [('miss_chain1','chain1'),('q_path','chain2 (기본)'),('miss_chain2','chain2 (반복)'),('miss_chain4','chain4'),('miss_star3','star3'),('miss_tree2x2','tree2x2')]:
   r=ix.get((group,n))
   if r:add(f'| {m} | {b} | {label} | {r["boundary_excluded_al"]:.3f} | {ci(r.get("delta_clean_al_ci95"))} | {r["al_by_source"][0]:.3f} | {r["p1_hit"]:.3f}/{r["p2_hit"]:.3f} | {r["later_pass_tps_mean"] or r["tps_mean"]:.1f} |')
  a=ix.get((group,'miss_tree2x2'));ref=ix.get((group,'miss_chain4'))
  if a and ref and a['cells']==ref['cells']:
   aa=np.array(a['per_question']);bb=np.array(ref['per_question']);samples=np.random.default_rng(4342).integers(0,len(aa),(2000,len(aa)))
   av=aa[samples].sum(1);bv=bb[samples].sum(1)
   delta=av[:,3]/av[:,4]-bv[:,3]/bv[:,4]
   contrasts.append(dict(group=group,candidate='miss_tree2x2',reference='miss_chain4',delta_clean_al=a['boundary_excluded_al']-ref['boundary_excluded_al'],relative_pct=100*(a['boundary_excluded_al']/ref['boundary_excluded_al']-1),ci95=np.quantile(delta,[.025,.975]).tolist(),tps_a=a['later_pass_tps_mean'] or a['tps_mean'],tps_b=ref['later_pass_tps_mean'] or ref['tps_mean']))
add();add('동일 4-node 비교: tree2x2는 forward2·깊이2, chain4는 forward4·깊이4.');add()
table(['모델/B','tree2x2 상대 AL*','ΔAL* CI','tree2x2 / chain4 TPS'])
for r in contrasts:add(f'| {r["group"]} | {r["relative_pct"]:+.2f}% | {ci(r["ci95"])} | {r["tps_a"]:.1f} / {r["tps_b"]:.1f} |')
(HERE/'MATCHED_NODE_CONTRASTS.json').write_text(json.dumps(contrasts,indent=2))
add();add('## 4. 같은 tree 점수의 실행 최적화');add()
table(['모델/B','설정','AL*','pass별 TPS','baseline과 같은 출력 질문 수/pass'])
for m in ['llama2','llama3']:
 for group,names,refname in [(m+'_optimized',['baseline','fused','fused_bulk','fused_bulk_parallel','short_k2_1_fused_bulk','short_k2_1_miss2_fused_bulk','combined_chain4'],'baseline'),(m+'_b1',['q_path','optimized_q_path'],'q_path'),(m+'_replica',['baseline','fused_bulk'],'baseline')]:
  ref=ix.get((group,refname)); refdata=json.load(open(HERE/ref['path'])) if ref else None
  for n in names:
   r=ix.get((group,n))
   if not r:continue
   data=json.load(open(HERE/r['path']));same=[]
   if refdata:
    for c,rc in zip(data['cells'],refdata['cells']):same.append(sum(a['token_ids']==b['token_ids'] for a,b in zip(c['outputs'],rc['outputs'])))
   add(f'| {group} | {n} | {r["boundary_excluded_al"]:.3f} | '+', '.join(f'{x:.1f}' for x in r['tps'])+' | '+', '.join(map(str,same))+' |')
add();add('`short_k2_1`은 phase 예산도 바뀌고, `combined_chain4`는 reach+gain+frontier와 miss chain4를 함께 적용한다. 나머지는 동일 q-path·K4/2이며 출력 동등성도 별도로 대조한다. T>0에서 정책/depth가 바뀌면 RNG 소비 순서도 바뀌므로 short/combined arm의 낮은 동일출력 비율은 losslessness 실패를 뜻하지 않는다. Bitwise parity 기준은 같은 알고리즘의 kernel 최적화 arm에 적용한다. TPS pass1에는 처음 만나는 graph bucket capture/새 prompt 처리 비용이 더 포함될 수 있다. `replica`는 다른 GPU pair 및 draft seed1/target3030·3031의 fresh-engine 확인이며 primary pair의 절대 TPS와 합치지 않는다.');add()
add('## 5. 경계 제외 및 batch barrier');add()
table(['모델/B','cap 도달 요청/pass','제외 event','구 counter 초과계수','full-B all-hit','전체 TPS/pass','경계 step 제외 TPS/pass'])
for m in ['llama2','llama3']:
 for suf in ['b1','full']:
  r=ix.get((m+'_'+suf,'q_path'))
  if r:add(f'| {m}_{suf} | {r["capped_requests"]} | {r["boundary_excluded_events"]} | {r["old_counter_overcount_pct"]:.2f}% | {r["full_batch_all_hit_fraction"]:.3f} | '+', '.join(f'{x:.1f}' for x in r['tps'])+' | '+', '.join(f'{x:.1f}' for x in r['boundary_excluded_step_tps'])+' |')
add();add('구 counter 초과계수는 이번 run의 accepted/emitted 차이다. 옛 논문 TPS가 같은 비율로 틀렸다는 추정에 사용할 수 없다. All-hit는 miss JIT가 없다는 뜻이며 P1/P2가 시간 안에 끝났다는 뜻은 아니다.');add()
add('## 6. 실제 full-active-batch steady profile');add()
table(['profile','step 수','P1/P2 ms (중앙값)','P1→proxy 여유 ms','P2→ready 여유 ms','P1 정시율','P2 ready 정시율'])
for r in json.loads((HERE/'timeline.json').read_text()):
 g=r['groups']['full_batch']
 if not g['n']:continue
 met=g['metrics'];p=lambda k:met[k]['p50']
 add(f'| {r["name"]} | {g["n"]} | {p("p1_ms"):.2f}/{p("p2_ms"):.2f} | {p("p1_before_proxy_ms"):+.2f} | {p("p2_before_ready_ms"):+.2f} | {g["p1_hidden_rate"]*100:.1f}% | {g["p2_hidden_rate"]*100:.1f}% |')
add();add('음수 여유는 마감 초과다. B1 timing도 forced-unified 경로다. 작은 16-question B8 profile의 적은 full-B 표본은 보조 자료다. B8 주 분석은 `bulk0/bulk1/fused/k2_1_bulk1`의 48-question profile이며 profile TPS는 성능 비교에 사용하지 않는다.')
(HERE/'TABLES.md').write_text('\n'.join(lines)+'\n')
print('Tables written; matched-node contrasts',len(contrasts))
