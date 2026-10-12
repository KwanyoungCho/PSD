"""Rebuild the review report from machine-readable measurements."""
import json,statistics as st
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'MODEL_RESULTS.json').read_text())
full=[r for r in rows if r['kind']=='full' and r.get('family','tree')=='tree'];profiles=[r for r in rows if r['kind']=='profile' and r.get('family','tree')=='tree'];chains=[r for r in rows if r['kind']=='profile' and r.get('family')=='chain']
greedy=[r for r in rows if r.get('family')=='greedy']
complete=len(full)==24 and len(profiles)==26 and len(chains)==24 and len(greedy)==16
fmt=lambda x:f'{x:.3f}'
lines=['# Round6: 승인 항목 반영 및 verification/KV 검증','',
      f'상태: {"계획한 verification 모델 실험 완료" if complete else "실험 진행 중"} — tree profile {len(profiles)}/26, chain profile {len(chains)}/24, full480 T.7 {len(full)}/24, greedy prototype {len(greedy)}/16. 분석 시작 기준 commit `2fe4596`, branch `feat/duet-mlsys-coverage`.','',
      '## 검토 결론과 승인 범위','',
      '- 승인 반영: 실제 온도 후보 점수, G>M 정확성 수정, resident/prefill B 상한, 동일 int64 payload wire. 원래 있던 구현을 논문 브랜치에 이번에 병합한 것은 아니다.',
      '- 새 집계: cap/clip 발생 batch-step과 실제 활성 요청 수가 nominal B 미만인 step을 제외. TPS의 token/time, AL, hit는 같은 step 모집단이다. 나머지 요청은 끝까지 생성한다. 원시 TPS와 제외 수를 함께 보관한다.',
      '- Verification과 KV 관련 변경의 **채택 승인은 아직 사용자 검토 대상**이다. 이번에는 진단용 계측 및 benchmark를 추가했고 새로운 KV promotion 정책이나 자동 경로 선택을 도입하지 않았다.',
      '- 고정 입력에서 chain fast 검증은 모든 측정 조건에서 빨랐다. Tree graph는 unified eager 대비 빨랐지만 T0 B1/B2에서는 기존 순차 greedy walk가 더 빨랐다. 단일 구현의 보편적 우위를 주장하지 않는다.',
      '- 공통 prefix/preemption 수정은 예전에 제거한 draft KV promotion과 별개다. 논문 기준 `a82f7d2`에도 tree accepted-path stage/restore가 있다. B>1은 이를 요청 ID 기준으로 확장한 것이다.','',
      '## 집계 정의','',
      r'$S=\{s: B_s=B,\ \neg\mathrm{cap}_s,\ \neg\mathrm{clip}_s\}$, '
      r'$\mathrm{TPS}^{*}=\sum_{s\in S}\mathrm{emitted}_s/\sum_{s\in S}\mathrm{seconds}_s$.','',
      'AL은 recovery를 포함한 요청별 accepted length, hit는 요청별 cache hit이며 S에 포함된 모든 요청 event를 사용한다. 제외 조건의 합집합으로 처리해 cap과 partial-B가 겹쳐도 두 번 빼지 않는다. EOS로 suffix가 잘린 경우도 clip으로 제외한다. EOS 자체가 무조건 제외 사유인 것은 아니다.','',
      '`summary.performance`가 이 기준의 주 성능이다. 기존 `decode_tps`, `end_to_end_tps`, `boundary_excluded_*`는 재현을 위해 그대로 남겼다. TPS*는 prefill과 tail을 포함하는 전체 serving throughput과 다르다. Full480 표는 첫480 pass 이후 두 번째480 pass만 사용하고, 전체 raw 파일에는 두 pass 모두 있다.','',
      '## 고정 실험 설정','',
      '| 항목 | 값 |','|---|---|',
      '| 모델 | dense LayerSkip-Llama2-7B + AMD-Llama-135m; dense LayerSkip-Llama3-8B + Qwama-0.5B-Instruct |',
      '| GPU / TP | RTX4090 24GiB ×2 per run, target TP1 / draft1. Profile PIX2,3와4,5; B1 full L2 NODE0,1 / L3 PIX6,7. 동일 모델의 각 비교 내에서는 같은 pair 사용 |',
      '| Algorithm | K1=4, K2=2; exit L2=21/L3=16; P1/P2 G=M=8/4; C=3; P1 roots2, draft/proxy fanout2/1 |',
      '| Root / tree | residual, topm, overlap mix0, q_path, actual T=.7. 정책 개선 효과와 실행 최적화를 혼합하지 않음 |',
      '| Execution | TREE_EXEC1, FUSED0, BULK0, PARALLEL_INSERT0, proxy stream0, packed tree0. graph0=accept graph OFF/trim0; graph1=ON/trim0; trim1=ON/trim1 |',
      '| Profile | 고정48 입력, B1/2/4/8, 출력96, warm pass8100/측정8101. B1에만 legacy 경로 추가 |',
      '| Full | 저장된6개 group ×80 = 전체480 입력, B1/B8, 출력128, 입력 상한512, 독립 process2회. 각 회에서480 warm+480 measured. 두 번째 반복은 arm 순서 역전 |',
      '| Correctness | 기존 GPU regression300개, 새 집계 unit test3개. Micro chain32/tree64 조건; tree trim0/1 acceptance ladder 확률·residual bitwise 비교 |','',
      'Legacy B1과 unified B1은 별도 실행기다. 여기의 legacy는 현재 branch에 남아 있는 기존 B1 경로로, 승인된 온도·공통 수정 등이 포함된다. `a82f7d2` checkout 전체와의 전후 재실행은 아니다. graph0→graph1은 같은 알고리즘이지만 RNG 소비 방식이 달라 생성 출력이 동일한 paired counterfactual이 아니다. 고정 logits micro와 반복 full 측정을 함께 본다. 원 논문의70B/Blackwell TP2 재현 실험은 아니다.','',
      '이번 설정은 verification 변경의 영향을 분리하기 위한 고정 대조다. Round5의 개선 root/위치/tree 정책을 다시 튜닝한 최선의 DUET 설정이나 SSD 대비 성능 수치로 해석하지 않는다. SSD를 새로 실행한 비교도 아니다.','',
      '## Chain/tree acceptance 고정 입력 측정','',
      'BF16 synthetic logits, 단위 ms. 각 조건5회×20호출, 순서 교차, GPU 동기화 후 wall time. 최종 CPU 결과 전송 포함, target forward/통신 제외. 아래는 vocab32k/128256, 노드4/8, tree C1/3 조건별 median의 평균이다. 실제 workload의 shape 가중 평균은 아니다.','',
      '| 종류 | T | B | 기준 | 최적화 | ms 전→후 | 변화 |','|---|---:|---:|---|---|---:|---:|']
micro=json.loads((HERE/'MICRO.json').read_text())
for kind in ('chain','tree'):
 for t in (.7,0):
  for b in (1,2,4,8):
   cs=[c for c in micro[kind]['cases'] if c['temperature']==t and c['batch']==b]
   before='general' if kind=='chain' else 'eager';after='fast' if kind=='chain' else 'graph_trim1'
   x=st.mean(c['median_ms'][before] for c in cs);y=st.mean(c['median_ms'][after] for c in cs)
   lines.append(f'| {kind} | {t} | {b} | {before} | {after} | {x:.3f}→{y:.3f} | {(y/x-1)*100:+.1f}% |')
lines += ['', 'Greedy tree의 serial legacy walk는 B1 약0.038ms/B2 약0.075ms로 graph 약0.124/0.127ms보다 빠르다. 따라서 tree graph가 기존 B1 모든 경로보다 빠르다고 쓰면 안 된다. T0는 draft 점수 기반 확률적 p/q 검증이 아닌 target argmax 경로 검증이다.','',
          '![Acceptance micro](figs/01_acceptance_micro.png)','',
          '## Full-model verification 진단','',
          '48개 입력의 계측 실행. 같은 full-B/cap 제외 기준에 더해 첫19 step, graph capture step 및 다음 step을 제외한 steady 진단이다. 이 추가 제거는 아래 진단 표에만 적용하며 full TPS*에는 적용하지 않는다. Verification은 기존 `target_verify_times`의 host wall 범위(입력 준비·target 실행·accept 결과 CPU 전달·KV commit dispatch 포함, draft 대기 제외). Acceptance 및 KV 열은 CUDA event span으로, 서로 다른 timing scope다.','',
          '| Model | B | 경로 | n | Verify wall mean ms | Accept span mean ms | Restore | Stage | Glue forward |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
for r in profiles:
 s=r['steady_profile'];m=s['gpu_span_ms'];arm=r['name'].split('_')[2];v=lambda k:f"{m[k]['mean']:.3f}" if k in m else '—'
 lines.append('| '+' | '.join([r['name'].split('_')[0],str(r['batch']),arm,str(s['steps']),fmt(s['verify_ms']['mean']),v('batch_tree_accept'),v('batch_kv_restore'),v('batch_kv_stage'),v('batch_glue_forward')])+' |')
lines += ['', '**기존 논문 B1 경로와의 직접 비교**: 아래는 같은48개 입력 설정에서 legacy B1과 unified+trim을 대조한 전체 verification 시간이다. 이후 full480 표의 graph0 기준 개선율과 구분한다. 이48입력 결과를 원 논문70B의 전체 데이터 TPS 개선율로 옮겨 쓰지 않는다.', '',
          '| Model | Legacy B1 ms | Unified+trim B1 ms | 감소율 |', '|---|---:|---:|---:|']
for model in ('llama2','llama3'):
 cs={r['name'].split('_')[2]:r['steady_profile']['verify_ms']['mean'] for r in profiles
     if r['name'].startswith(f'{model}_b1_')}
 if 'legacy' in cs and 'trim1' in cs:
  lines.append(f'| {model} | {cs["legacy"]:.3f} | {cs["trim1"]:.3f} | {(1-cs["trim1"]/cs["legacy"])*100:.2f}% |')
lines += ['', '### Chain full-model 후속 대조', '',
          '같은 두 모델과 K4/2·exit21/16·T.7, tree OFF. General(packed OFF/fast OFF), fast(packed OFF/fast ON), packed(packed ON/fast ON). 48입력, 첫pass8300/측정8301, 출력96, 시작 draft seed73. 모든 arm에서 proxy graph는 켜서 acceptance/packing 변경을 분리한다. 이 표도 steady profile latency이며 full480 TPS가 아니다.', '',
          '| Model | B | 경로 | n | Verify wall mean ms | Chain accept span ms |',
          '|---|---:|---|---:|---:|---:|']
for r in chains:
 s=r['steady_profile'];m=s['gpu_span_ms'];arm=r['name'].split('_')[3]
 lines.append(f'| {r["name"].split("_")[0]} | {r["batch"]} | {arm} | {s["steps"]} | {s["verify_ms"]["mean"]:.3f} | {m["chain_accept"]["mean"]:.3f} |')
if chains:lines += ['', '![Chain verification](figs/05_chain_verification.png)']
lines += ['', '| Model | B | General→fast verify ms | 감소율 | Fast→packed verify ms | 추가 감소율 |',
          '|---|---:|---:|---:|---:|---:|']
for model in ('llama2','llama3'):
 for b in (1,2,4,8):
  cs={r['name'].split('_')[3]:r['steady_profile']['verify_ms']['mean'] for r in chains
      if r['name'].startswith(f'{model}_chain_b{b}_')}
  if set(cs)!=set(('general','fast','packed')):continue
  x,y,z=(cs[a] for a in ('general','fast','packed'))
  lines.append(f'| {model} | {b} | {x:.3f}→{y:.3f} | {(1-y/x)*100:+.2f}% | {y:.3f}→{z:.3f} | {(1-z/y)*100:+.2f}% |')
lines += ['', 'Packed는 짧은 proposal의 padding을 줄이는 추가 경로이며 fast acceptance와 별도다. B1에서는 기존 dense 경로를 사용하므로 packed arm과 fast arm 차이는 packing 효과가 아니다. Query 절감보다 입력 준비·graph bucket 비용이 클 수 있어 B2 이상도 측정 결과를 확인한다.']
chain_lookup={r['name']:r for r in chains};chain_parity=[]
for name,r in chain_lookup.items():
 if '_general_' in name and name.replace('_general_','_fast_') in chain_lookup:
  q=chain_lookup[name.replace('_general_','_fast_')]
  chain_parity.append(r['output_sha256']==q['output_sha256'])
if chain_parity:lines += ['', f'General→fast의 측정 pass 출력은 {sum(chain_parity)}/{len(chain_parity)} model/B 대조에서 모두 같았다. Warm pass까지 포함한 검사는 AUDIT.json의 chain_fast_output_controls에 있다. Fast→packed는 계산 shape가 바뀌며 여러 B>1 조건에서 생성 경로도 달랐다. Packed의 시간 차이를 완전히 같은 token 궤적의 연산 차이로 해석하지 않는다.']
lines += ['', 'Legacy의 빈 칸은 비용0이 아니라 해당 batched label이 없다는 뜻이다. Legacy `tree_kv_restore` 등은 MODEL_RESULTS.json의 전체 label 통계를 본다. 상세 host dispatch와 CUDA span을 모두 보관한다.','',
          '![Model verification](figs/02_model_verification.png)','',
          '## Full480: 프로파일링 OFF','',
          '각 셀은 독립 process 반복들의 평균 [최소, 최대]. 반복2개로 유의성을 확정하지 않는다. AL/hit/TPS*는 같은 full-B/non-boundary step들만 반영한다. Raw TPS는 tail과 terminal step을 포함한다.','',
          '| Model | B | 경로 | 반복 | AL* | Hit* % | TPS* [min,max] | Raw TPS | Verify ms [min,max] | 제외 step % |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for model in ('llama2','llama3'):
 for b in (1,8):
  for arm in ('graph0','graph1','trim1'):
   cs=[r for r in full if r['name'].startswith(f'{model}_b{b}_{arm}_')]
   if not cs:continue
   pf=[r['performance'] for r in cs];t=[p['tps'] for p in pf];v=[r['verify_ms']['mean'] for r in cs]
   lines.append(f'| {model} | {b} | {arm} | {len(cs)} | {st.mean(p["al"] for p in pf):.3f} | {st.mean(p["cache_hit"] for p in pf)*100:.2f} | {st.mean(t):.2f} [{min(t):.2f},{max(t):.2f}] | {st.mean(r["raw_decode_tps"] for r in cs):.2f} | {st.mean(v):.3f} [{min(v):.3f},{max(v):.3f}] | {st.mean(p["excluded_steps"]/p["total_steps"] for p in pf)*100:.2f} |')
if full:lines+=['','![Full dataset](figs/04_full_dataset.png)']
comparisons=json.loads((HERE/'COMPARISONS.json').read_text())
lines += ['', '| Model | B | 비교 (graph0 기준) | 반복 | Verify 평균 변화 | TPS* 평균 변화 [반복 범위] |',
          '|---|---:|---|---:|---:|---:|']
for c in comparisons['summary']:
 lo,hi=c['tps_change_percent_range']
 lines.append(f'| {c["model"]} | {c["batch"]} | {c["arm"]} | {c["repetitions"]} | {c["verify_change_percent_mean"]:+.2f}% | {c["tps_change_percent_mean"]:+.2f}% [{lo:+.2f},{hi:+.2f}] |')
lines += ['', '변화율은 각 반복 내 같은 seed/GPU pair 대조의 변화율을 구한 뒤 평균했다. Graph의 효과와 trim의 추가 효과는 분리해야 하며, graph1↔trim1의 출력 일치 및 상대 변화는 COMPARISONS.json의 trim_controls에 기록한다.']
lines += ['', '### TPS가 verification 시간만 따라가지 않는 이유', '',
          r'같은 B를 유지하는 포함 step에서는 $TPS^*=B\times AL^*/\overline{t}_{step}$이다. 아래의 나머지 시간은 step wall minus verification wall이며 draft 응답 대기, scheduler, 후처리 등을 포함한다. 특정 원인 하나로 전부 해석하지 않는다.', '',
          '| Run | Step mean ms | Verification mean ms | 나머지 mean ms |',
          '|---|---:|---:|---:|']
for r in full:
 lines.append(f'| {r["name"]} | {r["step_ms_mean"]:.3f} | {r["verify_ms"]["mean"]:.3f} | {r["nonverification_ms_mean"]:.3f} |')
lines += ['', 'L2 B8의 첫 반복에서는 graph0→graph1에서 verification25.559→23.957ms, 나머지7.684→9.515ms로 움직였다. 전체 step33.243→33.472ms와 AL2.137→2.127이 함께 반영되어 TPS514.19→508.31이었다. Profile에서도 draft P1/P2가 긴 구성임을 확인한다. 다른 생성 궤적과 반복 변동이 있으므로 이 차이만으로 graph가 TPS를 인과적으로 악화시킨다고 확정하지 않는다.', '',
          '| Profile | Draft P1 mean ms | Draft P2 mean ms | Target spec wait mean ms |',
          '|---|---:|---:|---:|']
for r in profiles:
 if r['batch']!=8 or '_legacy_' in r['name']:continue
 m=r['steady_profile']['gpu_span_ms']
 lines.append(f'| {r["name"]} | {m["batch_p1_total"]["mean"]:.3f} | {m["batch_p2_total"]["mean"]:.3f} | {m["target_spec_wait"]["mean"]:.3f} |')

lines += ['', '## Greedy tree: 추가 실험용 개선', '',
          '32개 fixed-input 조건에서 target의 모든 문맥별 argmax ID를 한 번에 CPU로 옮기고, 이미 CPU에 있는 각 요청의 parent/token 정보로 greedy 경로를 따라갔다. Graph/reference와 path·recovery·terminal이 모두 일치했다. 생산 기본 경로에는 반영하지 않았다.', '',
          '실제 모델 대조는 benchmark adapter로만 적용한다. `mlsys_review_greedy_hook.py`가 현재 verifier에 CPU metadata 저장 한 줄을 넣고 greedy replay만 바꾼다. 원래 graph capture는 양쪽 모두 수행하므로 warmup/예약 메모리는 유지한다. T>0 구현은 그대로이며 probability-based 예산 계산도 바꾸지 않는다. Adapter0/1을 동일 모델·GPU pair·K4/2·exit21/16·C3·T0·출력64·seed8400/8401·draft74에서 비교한다.48입력 profile과 full480 두 pass(뒤 pass 측정)를 각각 실행한다. 새로운 production 선택 기준이나 파라미터를 추가한 결과가 아니다.', '',
          '| Model | B | Scope | Host | AL* | Hit* % | TPS* | Verify wall mean ms |',
          '|---|---:|---|---:|---:|---:|---:|---:|']
for r in greedy:
 p=r['performance'];v=r['steady_profile']['verify_ms']['mean'] if r['kind']=='profile' else r['verify_ms']['mean']
 lines.append(f'| {r["name"].split("_")[0]} | {r["batch"]} | {r["kind"]} | {r["env"].get("SSD_REVIEW_GREEDY_HOST")} | {p["al"]:.3f} | {100*p["cache_hit"]:.2f} | {p["tps"]:.2f} | {v:.3f} |')
lines += ['', 'Profile 행의 TPS는 계측 overhead가 포함되므로 full 행과 직접 비교하지 않는다. Full-model output equality와 상대 latency/TPS는 GREEDY_COMPARISONS.json에 보관한다. 이 대조는 T0, B1/B8, 두 dense 모델과 명시된 길이 범위의 근거다. Micro에서 batch별 graph→host는 B1 .137→.037ms, B2 .139→.038ms, B4 .141→.039ms, B8 .133→.041ms였다. 미시적인 약0.1ms 단축을 큰 TPS 향상으로 과장하지 않는다.', '']
lines += ['T0의 proposal/verification은 argmax다. 기존 `root_policy.probabilities`는 T0에서도 예산 순위 계산에 필요한 soft score를 유지하며, 이 실험은 그 규칙을 바꾸지 않는다. T>0에서 실제 온도로 확률을 계산하는 승인 항목과, greedy의 보조 점수 정의는 구분한다.', '']
greedy_comparisons=json.loads((HERE/'GREEDY_COMPARISONS.json').read_text())
lines += ['| Model | B | Scope | Warm/측정 출력 동일 | AL/hit/집계 동일 | Verify 변화 | TPS* 변화 |',
          '|---|---:|---|---|---|---:|---:|']
for c in greedy_comparisons['model_pairs']:
 lines.append(f'| {c["model"]} | {c["batch"]} | {c["kind"]} | {c["both_passes_equal"]} | {c["same_population_and_al_hit"]} | {c["verify_change_percent"]:+.2f}% | {c["tps_change_percent"]:+.2f}% |')
if greedy_comparisons['model_pairs']:lines += ['', '![Greedy full-model prototype](figs/06_greedy_prototype.png)', '']
gf=[c for c in greedy_comparisons['model_pairs'] if c['kind']=='full']
if gf:
 lines += [f'Full480에서 관측한 TPS 변화 범위는 {min(c["tps_change_percent"] for c in gf):+.2f}–{max(c["tps_change_percent"] for c in gf):+.2f}%다. Model/B별 독립 process 대조1회이므로 이 작은 차이의 통계적 우위를 확정하지 않는다. Micro에서의 큰 상대 개선율만으로 기본 구현 교체를 정당화하지 않으며, 실험용 후보로 남긴다.', '']
lines += ['', '## KV: 무엇을 비교했고 무엇을 바꾸지 않았는가','',
          '1. **공통 KV/요청 상태 수정**: SGL paged prefix prefill에서 실제 page table/요청별 길이를 읽기, 완전히 cached된 block-aligned prompt에서 마지막 query1개 유지, preemption 후 원래 출력 경계/남은 예산 유지, 완성된 block의 정확한 page에 hash 기록. 논리 주소를 바로 읽도록 하는 수정이며 전체 KV를 연속 공간으로 재조립하는 정책이 아니다.',
          '2. **기존 tree stage/restore**: target 검증 후 수락된 경로를 canonical prefix 위치에 반영하고, draft glue 결과를 저장했다가 다음 요청의 수락 경로에 맞춰 복원한다. Paper base `a82f7d2`의 `draft_runner.py`에도 `_tree_staged_kv`와 `tree_kv_restore`가 있다. B>1은 요청별 snapshot과 현재 page table을 사용한다.',
          '3. **제거된 KV promotion**: draft 후보 KV를 재사용해 confirmed-chain glue 전체 forward를 gather+tip1 forward로 대체하려던 별도 최적화. 과거20입력/out256 실험에서 Tdraft44.63→45.25ms, TPS76.06→76.49였고 비용 이득이 없어 제거했다. 당시 원래 build 시점 page table을 보존해야 했던 함정도 기록돼 있다. 이번에 되살리지 않았다.','',
          '**Glue의 대상도 구분해야 한다.** 현재 tree serving의 glue는 확정된 과거 prefix 뒤에 `[recovery + 현재 응답 tree 노드]`를 ancestor mask로 한 번 계산한다. Target이 그 tree를 검증하는 동안 다음 후보를 준비해야 하므로, 아직 확정되지 않은 여러 outcome context가 필요하다. 다음 요청에서 이전 tree의 수락 경로가 확정되면 그 경로만 canonical prefix로 복원한다. 과거 chain glue를 gather+tip1로 대체한 실험과 현재 branching-tree glue의 전체 대체는 같은 작업이 아니다.', '',
          '과거 근거: [KV promotion 설계](../../../ssd/docs/duet/11-kv-promo-design.md), [당시 결과](../../../ssd/experiments/proxy_async_overlap/kv_promo/RESULTS.md).','',
          '### 현재 stage/restore/glue 비용','',
          '![Existing KV cost](figs/03_existing_kv_cost.png)','',
          '전체 verification과 draft glue는 서로 다른 GPU/phase 구간이다. 위 비용을 단순 합산해 end-to-end TPS 증가량으로 바꾸지 않는다.','',
          '### 유지된 KV 복사 vs confirmed-chain 재계산 진단','',
          '실제 draft 모델, B1/2/4/8, context127/511, C1N8/C3N8/C3N16, 연속/불연속 물리 page. 조건별15회 순서 교차. 각 treatment 직전 원 KV 복원은 timing 밖이다. Snapshot clone+accepted path scatter+tip1 forward와 confirmed-chain one-forward를 비교한다. 이는 단일 유효 tree KV를 이미 보유하고 그중 한 경로를 택한 낙관적 micro이다. 실제 tree glue의 모든 outcome별 fork logits 및 아직 KV가 없는 leaf 계산을 대체하는 serving A/B가 아니며, cache 전체 보존/통신 비용도 없다.','',
          '| Draft / dtype | B | C | cases | copy ms | copy+tip ms | chain glue ms | max TV | argmax 불일치 cases |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in json.loads((HERE/'KV_RESULTS.json').read_text()):
 name='AMD FP16' if r['source'].startswith('amd') else ('Qwama BF16' if '_bf16_' in r['source'] else 'Qwama FP16 진단')
 m=r['mean_of_case_medians_ms'];lines.append(f'| {name} | {r["batch"]} | {r["siblings"]} | {r["cases"]} | {m["copy_only"]:.3f} | {m["promotion_plus_tip"]:.3f} | {m["confirmed_chain_glue"]:.3f} | {r["max_tv"]:.5f} | {r["argmax_disagree_cases"]} |')
lines += ['', '수치 진단: AMD48조건에서는 argmax 불일치0, 최대TV0.00211. Qwama BF16은6/48조건에서 하나 이상의 argmax 차이, 최대TV0.02903; FP16 진단은2/48, 최대TV0.00351. 이들은 합성 입력의 case 수이며 실제 데이터에서의 오류율 추정이 아니다.','',
          'Qwama BF16 분기형 tree(C3)의 KV는 **같은 query 폭으로 재계산한 경로 KV와 bitwise 일치**했다. 더 짧은 glue나 tip-only 계산에서는 shape-dependent 수치 차이가 있다. 이를 구조적 page 복원 오류와 구분할 근거는 있지만, 모든 shape/dtype에서 greedy 출력이 동일하다는 보장은 아니다. 진단의 TV1%는 문제 조사 trigger였고 losslessness 허용 오차가 아니다. `complete_diagnostic`를 정확성 통과로 읽지 않는다.','',
          '**여기서 비교한 것은 draft 모델의 KV/logits다.** Draft의 수치 차이나 argmax 차이만으로 target losslessness가 깨지는 것은 아니다. Draft proposal은 달라져도, 실제 생성에 사용한 q를 verifier에 정확히 전달하고 target의 조건부 p를 올바른 문맥에서 계산하면 SD의 분포 보존 조건은 유지된다. 이번 probe가 보장하지 않는 것은 복사/재계산의 bitwise 동일성 및 새 promotion serving 전체의 q·KV lifetime 계약이다. Target prefix cache가 다른 요청의 문맥을 읽는 오류와 구분해야 한다.', '',
          '### Prefix reuse full-model 측정','',
          '별도 AR 모델에서 첫8개 입력의 token열을 반복해 길이512/768/800으로 구성하고 B1/B8, greedy8출력, cold/cached 순서 교차3회. Cold는 완료된 요청의 hash lookup만 비우며, 두 arm 모두 수정된 정상 구현이다. 측정은 반복 prompt에 대한 prefill latency이고 일반 workload의 hit율/TPS가 아니다.','',
          '| Model | B | Prompt | Cold prefill ms | Cached prefill ms | Cached token count range | Output equality |','|---|---:|---:|---:|---:|---|---|']
for model in ('llama2','llama3'):
 p=HERE/f'{model}_prefix.json'
 if not p.exists():continue
 d=json.loads(p.read_text())
 if d['status']!='complete':
  lines.append(f'\n{model} prefix 상태: {d["status"]}; {d.get("error", "")}.');continue
 for b in (1,8):
  for length in (512,768,800):
   cs=[c for c in d['cases'] if c['batch']==b and c['length']==length]
   means={k:st.mean(c['arms'][k]['prefill_ms'] for c in cs) for k in ('cold','cached')}
   cached=[z['cached'] for c in cs for z in c['arms']['cached']['allocation']]
   lines.append(f'| {model} | {b} | {length} | {means["cold"]:.3f} | {means["cached"]:.3f} | {min(cached)}–{max(cached)} | {sum(c["greedy_tokens_equal"] for c in cs)}/{len(cs)} repeats |')
lines += ['', '## 검토 시 구분할 한계 / 남은 승인','',
          '- Verification: 고정-logits acceptance 단축과 전체 model verification/TPS 효과를 구분한다. B1 legacy→unified의 효과를 graph OFF→ON 비교로 대체하지 않는다. T0의 작은 batch는 경로 선택 개선 여지가 남는다. 이번에는 그 선택 정책을 자동 도입하지 않았다.',
          '- KV: 공통 정확성 수정과 KV promotion 채택은 별도다. 새 promotion의 전체 serving 성능·lifetime·수치 계약이 검증됐다고 주장하지 않는다. 기존 correct stage/restore를 제거한 “빠른” 경로도 만들지 않았다.',
          '- 데이터 범위: B2/B4는48입력 진단과 fixed-input micro; full480은 B1/B8이다. 주 비교 온도는 T=.7이며, T0의 추가 host prototype은 별도 절의 출력64·profile/full480 대조 범위다. 출력128, target7B/8B, TP1 결과를70B/TP2나 긴 문맥 전체에 일반화하지 않는다.',
          '- 승인된 집계는 조건부 비교다. 제외율이 큰 설정을 raw serving TPS보다 좋아 보이게 발표하지 않도록 원시 TPS와 제외 수를 같이 확인한다.',
          '- 결과 재현: `python archive_results.py --restore`(다른 서버), `python analyze.py`, `python compare.py`, `python compare_greedy.py`, `python make_figures.py`, `python make_report.py`, `python audit_results.py --require-complete`. 모델 실행 인자/환경은 각 plan.json과 raw 결과에 기록. [계획](PLAN.md), [실패 기록](FAILURES.md), [전체 모델 수치](MODEL_RESULTS.json), [micro 수치](MICRO.json), [KV 수치](KV_RESULTS.json).','']
lines += ['## 다른 서버에서 검토·재실행', '',
          '1. 채택 여부는 상위 MERGE_REVIEW.md 26절에서 확인한다. 이 결과를 가져오는 것과 논문 branch에 추론 코드를 병합하는 것은 별도다.',
          '2. `archive_results.py --restore`는 gzip 원본을 SHA256 검사 후 복원한다. 기존 raw가 다르면 덮어쓰지 않고 중단한다. 실패 실행도 보존한다.',
          '3. 같은 디렉터리에서 위 분석 명령들을 순서대로 실행한다. AUDIT.json은 매 pass의 실제 반환 token 합, 집계 모집단, B 상한, 두 pass 출력 대조, 외부 GPU 프로세스 유무를 검사한다.',
          '4. 모델 실행을 다시 할 때는 plan의 target/draft/prompts/profile 경로를 새 서버에 맞게 바꾼 사본을 사용한다. 원 plan과 원 raw를 덮어쓰지 않는다. 기존 `ssd/bench/mlsys_campaign.py --plan PLAN --directory NEW_DIR --gpus I,J --port PORT`로 실행한다. 각 모델의 비교 arm은 같은 GPU pair에서 순차 실행해야 한다.',
          '5. 원래 실행한 tree profile, B1/B8 full, chain profile, greedy prototype은 아래 plan에 있다. Full에는 repeat0/1의 역순 실행과 warm/측정 두 seed가 포함된다. Greedy model 대조는 benchmark module을 import해 해당 process의 replay만 대체하며 생산 소스 파일을 바꾸지 않는다.', '',
          '| Suite | Plan | 원 GPU pair L2 / L3 |', '|---|---|---|',
          '| Tree profile | `{model}_profile_plan.json` (실행 사본은 각 campaign의 plan.json) | 2,3 / 4,5 |',
          '| Tree full B1 | `{model}_b1_full_plan.json` | 0,1 / 6,7 |',
          '| Tree full B8 | `{model}_b8_full_plan.json` | 2,3 / 4,5 |',
          '| Chain profile | `{model}_chain_profile_plan.json` | 2,3 / 4,5 |',
          '| Greedy prototype | `{model}_greedy_review_plan.json` | 2,3 / 4,5 |', '',
          'MODEL_MANIFEST.json은 config/tokenizer의 SHA256과 weight shard의 크기·해결된 경로를 기록한다. 큰 weight 파일 전체의 SHA256을 계산한 manifest는 아니다. Raw 결과에는 실행 당시 Python 소스 SHA256도 있다. 기본 추론 파일은 측정 도중 바꾸지 않았으며, greedy host1의 메모리 내 adapter는 별도 실행 변경으로 표시했다.', '',
          '새 성능은 `summary.performance`에서 읽는다. 과거 `decode_tps`를 같은 뜻으로 바꾸거나 과거 그림 수치를 덮어쓰지 않는다. 오래된 AR raw에 actual B 정보가 없으면 정확한 tail 제외를 추정하지 않고 재실행해야 한다.', '']
(HERE/'REPORT.md').write_text('\n'.join(lines))
print('Wrote REPORT.md',len(full),len(profiles))
