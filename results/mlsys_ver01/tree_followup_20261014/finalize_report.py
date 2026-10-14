"""Render the single handover report after all declared validation jobs finish."""
from pathlib import Path
import json,sys,statistics
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chokwans99/duet-tree-20261014')
sys.path.insert(0,str(HERE.parents[2]/'ssd/bench'))
from mlsys_tree_results import compare

def read(name):
 j=json.loads((ROOT/name).read_text())
 if j.get('status')!='complete':raise RuntimeError(name+' incomplete')
 return j

def pair(a,b):return compare(read(a),read(b))
def fmt(x):return f'{x:.3f}'
def ci(v):return f'[{v[0]*100:+.2f}, {v[1]*100:+.2f}]%'

checks=[('Llama2 / 4090 / C3','candidate_full_l2/base.json','candidate_full_l2/candidate.json'),('Llama3 / 4090 / C2','candidate_full_l3/base.json','candidate_full_l3/candidate.json'),('Llama3 / 3090 / C3','remote/full_l3/base.json','remote/candidate_l3/tp1_candidate.json')]
full=[]
for label,a,b in checks:
 x,y=read(a),read(b);c=compare(x,y);s,t=x['summary'],y['summary']
 full.append(f"| {label} | {s['verify_ms']:.3f} → {t['verify_ms']:.3f} | {100*c['verify_change']:+.2f}% {ci(c['verify_change_ci95'])} | {s['tps']:.2f} → {t['tps']:.2f} | {s['al']:.4f} | {100*s['hit']:.2f}% | {c['equal_output_sequences']}/{c['output_sequences']} |")
mixed=[]
for label,a,b in [('4090, K2=2, miss≤1','mixed_full_local_tuned/tuned_wait.json','mixed_full_local_tuned/tuned_m1.json'),('4090, K2=4, miss≤2','mixed_full_local_long/long_miss_wait.json','mixed_full_local_long/long_miss_m2.json'),('3090, K2=2, miss≤1','remote/mixed_full_l2/tuned_wait.json','remote/mixed_full_l2/tuned_m1.json'),('3090, K2=4, miss≤2','remote/mixed_full_l2/long_miss_wait.json','remote/mixed_full_l2/long_miss_m2.json')]:
 x,y=read(a),read(b);c=compare(x,y);s,t=x['summary'],y['summary']
 mixed.append(f"| {label} | {s['al']:.3f} → {t['al']:.3f} | {s['step_ms']:.2f} → {t['step_ms']:.2f} | {s['tps']:.2f} → {t['tps']:.2f} | {100*c['tps_change']:+.2f}% {ci(c['tps_change_ci95'])} |")
time_rows=[]
for r in json.loads((HERE/'SELECTED_TIMING.json').read_text()):
 tag=Path(r['folder']).name.replace('_profile','');m=r['timing_medians']
 time_rows.append(f"| {tag} | {m['proxy_ready']:.2f} | {m['p1_end']:.2f} | {m['p2_end']:.2f} | {m['next_request']:.2f} | {m['p2_margin']:.2f} | {100*r['p2_late_fraction']:.1f}% |")
section=r'''
## 2026-10-14 tree 후속 최적화 — 최종 검증과 반영

### A. 이번 작업의 결론과 실제 반영 범위

- **채택**: (1) Draft P1/P2 입력 metadata 준비에서 작은 CPU torch tensor 반복 생성 제거, (2) target attention을 유지하면서 proxy·수락의 vocabulary 연산만 실제 context 수로 압축, (3) B>1·TP>1 tree에서 rank0 full LM head + 별도 stream 자동 사용.
- **함께 반영**: 설정된 batch/context/page/Q 버킷의 CUDA graph를 serving 전에 준비. 준비 시간을 decode 성능으로 계산하지 않는다. 그래프마다 큰 입력 버퍼를 따로 만들지 않고 최대 크기 버퍼의 prefix view를 재사용한다.
- Mixed-miss AR 및 target attention 자체의 전체 packed 실행은 기본 경로에 채택하지 않았다. 아래에 전체 데이터 결과와 이유를 남긴다. 사용자가 정할 새 packed/NumPy/stream 옵션은 추가하지 않았다. 기존 내부 debug 스위치는 실험 재현용으로 남는다.
- 새 runtime 반영 파일은 `config.py`, `engine/{llm_engine,model_runner}.py`, `engine/helpers/{batch_tree_inputs,batch_tree_verify,batch_tree_draft,batch_tree_forward,flat_tree_sampling,batch_tree_warmup}.py`의 **9개**다. 이 중 뒤 두 helper는 새 파일이다.
- 변경은 **MLSys-ver0.1 작업 트리**에 반영했다. HEAD는 f12d0f1이며 이번 추가분은 미커밋/미푸시다. 이전부터 있던 미커밋 승인 extension 및 실험 파일을 삭제하지 않았다. 이번 채택분은 별도 `ADOPTED_RUNTIME.patch`와 runtime hash로 구별한다.
- 논문 정본 `feat/duet-p2tree-g0@a82f7d2`는 그대로 보존했다. 이번 B>1 비용 대조의 before는 **f12d0f1 + 이미 검토한 extension의 동결본**이다. 원본 논문의 B1 결과와 바로 비교한 수치가 아니다. 기존 B1 기본 실행 경로는 그대로이며, B1 통합 실행기 실험은 `SSD_BATCHED_TREE=1`을 명시했다.

### B. Packed의 의미: KV를 합치는 것이 아니다

예: 요청 A는 query 5개, 요청 B는 query 2개이고, page table은 각각 `[5,9]`, `[2,18]`이다.

- 기존 padded query: `[A0,A1,A2,A3,A4]`, `[B0,B1,pad,pad,pad]` → 10개 위치.
- 논리 packed query: `[A0,A1,A2,A3,A4,B0,B1]` → 7개. Graph 버킷을 8개 단위로 맞추면 물리적 query는 8개가 될 수 있다.
- 각 query는 **자기 request ID, 자기 page table, 자기 prefix 길이와 RoPE 위치**를 유지한다. B의 query가 A의 KV를 읽지 않는다.
- Attention 대상은 `해당 요청의 prefix ∪ 해당 tree node의 조상`이다. 서로 다른 요청들이 prefix를 공유할 필요가 없다. Prefix 공유는 **한 요청 안의 tree 분기들 사이**에 있다.
- 구현은 전체 요청의 KV를 옆으로 합친 거대한 global mask를 만들지 않는다. Query별 request-local page 목록과 mask를 사용한다. Fixed-plan 후보에서는 query마다 page table을 반복해 plan을 고정했기 때문에, 기존 grouped attention의 KV 재사용 효율이 떨어질 수 있다.
- **채택한 경로는 target query 자체를 packed로 바꾸지 않는다.** Target attention/GEMM은 그대로 두고, 이미 계산한 logits 중 유효 context만 모아 proxy와 acceptance의 큰 `[contexts,V]` 계산량을 줄인다. Top-k의 작은 점수 배열만 request별 형태로 되돌린다.
- 균일한 길이로 줄일 행이 없으면 기존 dense 경로를 쓴다. 실제 B가 graph capacity와 다른 tail batch도 기존 경로다. 사용자 선택 옵션 없이 내부에서 결정한다.

![요청별 KV 소유권](tree_followup_20261014/PACKED_OWNERSHIP.png)

### C. 직접 측정한 비용 개선

서로 같은 입력을 번갈아 실행한 대조다. AL/cache hit 변화가 섞인 전체 TPS로 아래 항목을 판단하지 않았다.

| 변경 | 고정 조건과 측정 범위 | Before → after | 해석 |
|---|---|---|---|
| Draft 입력 준비 | 3090, B8, context 5/9/17; CPU 준비+GPU 입력 갱신 함수 | 약 1.69–1.80 → 1.30–1.40ms | 약 0.39–0.41ms 감소/호출. P1과 P2 각각 사용 |
| Proxy·수락 압축 | 4090, BF16, V128256, B8, N4/C2, Q40→32 | **2.561 → 1.878ms** | CPU topology, H2D, exit/final logits 각각의 gather, proxy, acceptance까지 포함. Target forward/통신 제외 |
| Proxy·수락 압축 | 같은 조건에서 N8/C3, Q72→48 | **5.951 → 3.442ms** | 같은 전체 처리 구간의 감소 |
| Full head + stream | Llama2, 4090, TP2/B8, 같은 실제 prefix/tree 입력, 2회 process | 17.562→17.271 / 17.779→17.381ms | target forward+proxy 구간 약 0.29–0.40ms 감소 |
| Full head + stream | Llama3, 3090, TP2/B8, 같은 입력, 2회 process | 42.619→39.991 / 42.598→39.952ms | 약 2.63–2.65ms, 약 6.2% 감소 |

- `probability_complete_component_r2.json`은 **exit와 final logits를 두 번 gather하는 비용까지 포함**한다. 앞선 single-pack 탐색 결과는 별도 보존하고 위 근거로 쓰지 않았다.
- GPU kernel만 비교한 BF16/N8/C3/Q72→48 대조: proxy 2.640→1.558ms, acceptance 2.489→1.492ms. 이 숫자는 전체 target latency가 아니다.
- Draft 전체 phase의 ABBA 계측도 했다. Llama2 B8에서 before의 P1 평균 8.127/8.156ms, NumPy 입력 적용 후 7.895/7.901ms. P2 평균 5.329/5.848→5.145/4.986ms였다. Target pre/post 중앙값은 약 10.785/6.48ms로 유지됐다. 계측 실행의 TPS는 성능 주장에 사용하지 않는다.
- Full head 복제 **단독**은 안정적인 개선이 없었다. 통신 없는 head와 별도 stream의 결합을 채택한다. Target과 draft의 GPU 간 병렬 실행은 모든 arm에서 원래 존재했다. 이번 stream은 **target GPU 내부에서 proxy와 남은 target layers를 겹치는 것**이다.
- B1에서도 TP2 component 대조는 개선: Llama2 10.529→10.304 / 10.559→10.353ms, Llama3 17.220→16.926 / 17.171→16.876ms. 그러나 이번 새 자동 기본값은 B>1 tree/TP>1에 적용하며, 기존 논문 B1 기본 경로를 바꾸지 않는다.
- TP1에서는 원래 head가 한 GPU에 있고 TP gather도 없다. TP1의 별도 stream 결과를 TP2 full-head 복제의 근거로 혼동하지 않는다.

![직접 component 대조](tree_followup_20261014/COMPONENTS.png)

### D. Full model / 전체 평가집합 / 통합 runtime 대조

- 양자화하지 않은 LayerSkip-Llama2-7B + AMD-Llama-135m(FP16 실행), LayerSkip-Llama3-8B + Qwama-0.5B-Instruct(BF16 실행).
- 저장소 평가집합 **480문항 전부**: mt_bench/translation/summarization/qa/math_reasoning/rag 각80. 원천 공개 데이터셋 전체를 뜻하지 않는다. 주요 A/B는 2 pass로 960개 출력, 독립 960문항은 아니다.
- T=0.7, input cap512, output cap64, max_model_len2048, B8. Warm prompts는 성능에서 제외. 모든 arm에서 동일한 `PAPER_COLD_PREFIX=1`.
- 사용자 승인대로 cap/clip step과 실제 B 미달 step은 토큰·시간·AL·hit에서 함께 제외한다. Graph 준비가 겹친 step도 통째로 제외한다. Raw 전체 step/출력은 보존한다.
- 95% 구간은 동일 prompt-group을 두 pass에 걸쳐 묶은 cluster bootstrap이다. GPU 환경 변동까지 포괄하는 구간은 아니다. `TELEMETRY.json`에 decode 구간 clock/온도/메모리와 외부 PID를 기록한다.

| 통합 A/B (B8) | verify ms | 변화, 95% 구간 | TPS (보조) | AL (동일) | hit (동일) | 같은 출력 |
|---|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(full)+r'''

- **Llama3/4090/C2는 전체 verify 향상을 확정하지 못했다. TPS도 −0.68%였다.** 이를 숨기거나 전체 throughput 개선이라고 부르지 않는다. C절의 고정 입력 직접 처리 비용은 줄었지만, 모든 설정의 전체 critical path가 같은 비율로 줄어드는 것은 아니다. 사용자 요청대로 component 비용을 채택 기준으로 삼았다.
- Llama2 전후 target GPU 평균 SM clock은 약2675.16/2674.43MHz, Llama3/4090은2722.86/2720.16MHz. 외부 프로세스는 관측되지 않았다. 작은 음수 결과를 clock 차이로 단정하지 않았다.
- Llama3/3090의 AL 중심 설정(K1/K2=3/2, exit19, N1/N2=6/4, roots1, C3)에서는 초기 adapter 대조도 수행: verify39.348→37.149ms(−5.59%), TPS359.41→375.53, 960/960 같은 출력. 이것은 위 최종 runtime/C3·2/2 설정의 결과와 구별한다.
- 초기 C2/3090 adapter는 933/960 출력만 같았다. 반면 최종 C2/4090 runtime은960/960 같았다. 모든 장비·설정에서 bitwise 동일하다는 일반 보장을 주장하지 않는다. Ordered-WOR law와 request별 RNG 형태는 독립적으로 검증했다.
- 실제 compact가 적용된 full-B step 비율은 Llama2 1761/1990, Llama3/4090/C2 476/1237, Llama3/3090/C3 242/1133이었다. 단일 kernel의 큰 개선률을 모든 step에 곱하지 않는다.
- B8/T0 Llama2는48/48 같은 출력. Llama3 TP2는 T0/T0.7 각각48/48 같았고, verify는47.289→45.019 / 50.867→47.406ms였다. 이 48문항 검사는 기능 통합 확인이며 full-dataset TPS 근거로 승격하지 않는다.

### E. 파라미터: 시간창으로 후보를 줄이고 근방을 확인

모델별 고정 `K1+K2`를 먼저 가정하지 않는다. 한 forward의 비용은 B, root 수, 해당 round의 query 폭, 문맥/KV 길이, 모델에 따라 달라진다.

```
P1_end    = glue_start + glue + D1(K1, roots, tree shape, B, context)
cache_end = max(P1_end, proxy_ready(exit, verify shape)) + D2(K2, roots, tree shape, B, context)
노출 지연 = max(0, cache_end - next_target_request)
```

1. Target pre/post, proxy 전달, draft glue/P1/P2를 별도 측정한다.
2. P1이 proxy보다 먼저 끝나는 것을 목표로 하고, P2가 다음 target handshake 시작 전에 끝나는 후보를 남긴다.
3. 두 phase의 합 자체가 시간창을 넘으면 exit만 움직여 해결할 수 없다. K, roots, node/width budget을 먼저 줄인다.
4. 남은 후보에서 exit·K·roots·node·sibling 수의 근방을 96문항으로 확인한다. 마지막에 480문항으로 AL/hit/latency를 함께 확인한다.
5. 96문항은 480개에서 균등 간격으로 뽑았다. 480문항 재평가와 일부가 겹치며 독립 holdout은 아니다. **전역 최적의 수학적 보장 또는 완성된 자동 autotuner는 아니다.** 측정된 시간 제약과 유한 근방 탐색의 결과다.

- 4090/Llama2 B1에서 6/4는 P1/P2 약5.6/4.0ms로 들어갔다. B8/roots2에서는 약23.4/13.1ms로 늘고 P2가 약7.8ms 늦어졌다.
- 3090/Llama3 B8의 초기4/2·N8/4·roots2는 P1/P2 약35.8/15.1ms로 P1부터 늦었다. 단순히 'P1을 길게' 하면 더 나빠지는 경우다.
- C2는 넓이만 줄이는 임의 trick이 아니다. 실제 sibling 단계 및 draft 폭을 줄이는 기존 tree parameter다. 같은3090 GPU4/5에서 C3→C2 full 대조는 TPS370.28→383.05, AL2.1502→2.1543, hit65.72→65.96%였다. 이는 시스템 변경과 분리한 parameter 대조다.

| 용도/장비 | B | K1/K2 | exit 코드 인덱스 | P1/P2 생성=검증 node 상한 | P1 roots/위치 | sibling C |
|---|---:|---:|---:|---:|---:|---:|
| Llama2/4090, 시간창 기준점 | 8 | 3/2 | 19 | 6/4 | 2 | 3 |
| Llama3/3090, 시간창 기준점 | 8 | 2/2 | 17 | 4/4 | 2 | 2 |
| Llama3/3090, AL 비교점 | 8 | 3/2 | 19 | 6/4 | 1 | 3 |
| Llama2/4090, 통합 B1 실행기 참고 | 1 | 8/4 | 21 | 16/8 | 2 | 3 |

- Exit 인덱스19는 20번째 transformer block 뒤다. 논문 표에 옮길 때 0-based 코드 값과 layer 개수를 구별한다.
- 동일한 고정 항목: target/draft 온도0.7, 기존 residual root/관측 위치식, P1 allocation dynamic, P2 selector dynamic, proxy threshold0.01, tree confidence threshold0.03. Root/위치 연구 수식을 이번 비용 대조에서 섞지 않았다.
- Llama2 B1의8/4는 통합 실행기 강제 실험에서480문항 기준 TPS110.63→116.24, AL1.989→2.166. 기존 논문 B1 기본 실행기의 튜닝 결과는 아니다.
- Llama2 긴 K2 비교점3/4·exit15·N6/8·roots1은 full4090에서 AL2.128이었다. 짧은 기준점의1.953보다 높지만 서로 다른 GPU pair로 측정한 두 TPS를 이용해 단일 최적을 확정하지 않는다. AL/latency 교환을 보여주는 비교점으로 남긴다.

최종 설정의 timeline 재검사(각 열은 별도 중앙값, target pre 시작=0ms):

| 설정 | proxy ready | P1 end | P2 end | 다음 request | P2 여유 | P2 늦은 비율 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(time_rows)+r'''

`selected_b1/b8`은4090/Llama2, `throughput/al_pareto`는3090/Llama3다. 작은 calibration 표본의 늦은 비율0%를 모든 요청의 deadline 보장으로 해석하지 않는다. AL 비교점은 P1 지연과 P2 tail 지연이 남아 시간창 기준점과 분리했다.

### F. Mixed-miss는 긴 K2까지 full로 확인했지만 기본 채택 근거가 없다

AR 처리의 이득은 'miss가 하나라도 있음'만으로 결정되지 않는다.

```
기다릴 때:     예상 토큰 A_wait,  시간 T_wait
AR로 넘길 때:  예상 토큰 A_AR,    시간 T_AR
AR가 유리한 조건: A_AR/T_AR > A_wait/T_wait
동치: A_wait - A_AR < A_AR * (T_wait - T_AR) / T_AR
```

Miss 수 m, miss당 AL, batched JIT 비용, target shape 변화가 필요하다. 이전 P2가 아직 실행 중인 대기는 추가 JIT를 생략해도 사라지지 않는다. 따라서 모든 모델에 공통인 'miss 1개/2개부터'라는 상수를 이번 데이터로 정할 수 없다.

Llama2, B8, 480문항×2. 각 행은 같은 GPU pair·같은 parameter·같은 시스템 최적화에서 wait와 AR만 바꾼 대조다. 4090은 최종 runtime,3090은 동일 기능 adapter 경로를 사용했다.

| 조건 | AL | step ms | TPS | TPS 변화,95% 구간 |
|---|---:|---:|---:|---|
'''+ '\n'.join(mixed)+r'''

- 4090의96문항 긴K2 탐색은 +7.55%였으나, **같은4090 full 대조는 −0.15%, 구간이0 포함**이었다. 작은 표본의 양수를 확정 개선으로 사용하지 않았다.
- 3090의 감소도 있으므로 기본 경로는 miss JIT를 유지한다. AR을 부정확한 방법이라고 판정한 것이 아니라, 이번 조건에서 확실한 처리량 이득이 없다는 판정이다.
- B>1 cache 통계는 request hit, all-hit step, any-miss step, mixed step, all-miss step, 실제 JIT step/request, AR bypass step/request, miss-count histogram을 모두 남겼다. Group 첫 step을 제외한 통계도 별도로 있다.
- 예: Llama3 C2/3090 wait는 request hit65.96%였지만 any-miss step94.06%였다. 높은 any-miss 비율만으로 AR가 유리하다는 결론은 나오지 않는다.

### G. 전체 packed target와 추가 계산 제거의 판정

- 고정 prefix/tree를 넣은 full-target component 대조: 4090/Llama2 half-short Q72→48은 padded 약19.07ms, packed 약19.16ms로 확실한 이득이 없었다. One-hit Q72→16은 약19.20→16.81ms였다.
- Llama3는 전체 packed 이득이 더 컸다. 4090 half-short 약24.24→21.99ms,3090 약41.1→31.6ms. 그러나 GEMM/attention shape 변화로 logits도 달라졌다. 3090 half-short는48개 context 중47개 argmax 일치, 최대 logit 차 약0.2344였다.
- Request 간 KV 격리가 깨진 결과는 아니다. 독립 dense attention과 페이지/조상 mask 대조는 통과했다. 그렇더라도 모든 모델에 공통인 latency 개선과 target 계산의 동일성을 동시에 확정하지 못해, 기본 target attention을 바꾸지 않았다.
- 실제 없는 sibling 단계 생략도 CPU120조건·CUDA 동일 seed 대조 및 component 측정을 했다. 다만 현재 full C3 trace는 대부분 최대 sibling3이 존재하고, 별도 C별 graph/buffer 추가가 필요하다. 단독 full-serving 개선을 확정하지 않은 추가 경로는 기본값에 넣지 않았다. C2 parameter의 이득과 혼동하지 않는다.
- CPU의 모든 작업이나 모든 동기화를 없앴다는 주장은 하지 않는다. 이번 반영 대상은 측정으로 개선을 확인한 입력 준비/확률 처리/stream 경로다.

### H. Graph 준비, 정확성, merge 시 확인할 계약

- Graph key는 tree 모양 전체가 아니라 batch capacity, context/node width, page bucket, compact Q다. Parent/mask/page table은 버퍼 값으로 갱신한다. 가능한 모양마다 새 graph를 만드는 구조가 아니다.
- Draft 준비는 `draft_ready` barrier 앞, target 준비는 engine init 완료 전에 한다. Global CUDA RNG와 draft arena generator 상태를 보존한다.
- 본문 full 대조3개 및 TP2/T0 통합 검사는 측정 중 신규 준비0건. Llama2 smoke의 준비는 draft120 graphs 약58.25초, target19 shapes 약24.24초였다. Llama3의2/2 설정은 draft80 graphs54.50초, target13 shapes19.36초였다. Startup·메모리 비용은 공짜가 아니다.
- Prompt265/777/1545/1909 tokens의 B4 검사: before는decode108step 중9step에서 준비6.475초가 겹쳤고, after는0건. 출력16/16 동일. 이 준비 시간을 제외하도록 했으므로 이를74% throughput 개선으로 인용하지 않는다.
- 마지막 경계 검토에서 첫 draft round의 footprint는 한 root의 max_nodes가 아니라 전체 context의 root 수라는 점을 반영했다. `max_model_len=2040` 같은 block 비정렬 상한도 포함하도록 page 범위를 산출한다. 기존2048 실험의 준비 bucket 집합은 바뀌지 않는다.
- 비정렬 상한2040·prompt1909/2020·B4 추가 full-model 검사:8/8 같은 출력, after decode 준비0건. Page 범위 CPU 검사2개(2010/2040/2048/4097 및 여러 root 폭) 통과.
- CUDA 회귀25개: 요청별 페이지/조상 attention, 별도 arena와 fused executor, KV 변경의 다른 요청 격리, ragged/빈 proposal, greedy/stochastic/WOR 소진, 입력 buffer parity, model/tokenizer 계약.
- Compact production 그래프는 FP16 V32000/BF16 V128256, C2/C3, 반복 Q 변경, T0/.3/1, 동일 seed240조건 통과. 앞선 all-FP16240조건 및 CPU12 law/isolation 조건도 별도 통과. 테스트 실행 수와 내부 parameter case 수를 합산해 독립 테스트 수로 부풀리지 않는다.
- `flat_tree_sampling.verify`는 node별 실제 draft law와 ordered sibling rejection을 유지한다. Root 후보 점수를 acceptance의 분모로 사용하지 않는다. T0은 target argmax 경로만 사용한다.
- **다른 서버에서 root/위치 수식을 merge할 때** dense의 `batch_tree_sampling.candidates`와 compact의 `flat_tree_sampling.candidates`를 함께 검토해야 한다. 현재 두 경로는 기존 residual/관측 위치식에 맞춰 있다. 새 e(1-q)/h_bar 정책을 한쪽에만 반영하면 shape에 따라 정책이 달라진다. `test_flat_tree_review.py` 및 production GPU parity 검사를 같이 갱신한다.
- Graph warmup은 설정한 최대 batch/문맥/예산에 대한 준비다. 이후 설정을 바꾸면 다시 준비해야 한다. 이번 메모리·성능 검증은4090/3090·7B/8B·B8 중심이며70B/Blackwell 결과로 대신하지 않는다.

### I. 증거와 재현

- 단일 설명 문서: 이 `REPORT.md`. 개별 계획/숫자: `tree_followup_20261014/ALL_SUMMARIES.json`, `SELECTED_TIMING.json`, `CALIBRATION_ANCHORS.json`, `PHASE_ABBA.json`, `TELEMETRY.json`, 각 campaign plan JSON.
- 그래프 재생성: `python results/mlsys_ver01/tree_followup_20261014/analyze.py --raw /data/chokwans99/duet-tree-20261014`.
- Raw archive·source snapshots·채택 patch·SHA256 목록: 같은 폴더의 `MANIFEST.json`. 최종 통합 A/B의 runtime은 snapshot과 대조한다. 초기 탐색 adapter는 작업 중 발전했으므로 개별 run의 adapter hash와 최종 source를 구별한다.
- 실험 실행기는 GPU telemetry 및 외부 PID를 기록하고 해당 실험 process group만 관리했다. 로컬4/5·6/7, 원격2/3·4/5·6/7를 사용했으며 원격 GPU0/1의 다른 프로세스는 유지했다.
- 실패 실행도 보존: 최초 assert 활성화 시 draft용 Config에 target exit-layer assert가 재적용되어 시작 실패; 기존 실험과 같은 `python -O`로 재실행. 원격 root filesystem 부족은 `/data2`의 build/tmp cache로 해결. Ephemeral port 충돌은 별도62xxx–65xxx port로 재실행. Replica+prewarm adapter의 Python closure 변수 충돌1건을 수정 후 재실행. 이러한 실패를 성능 표본에 넣지 않았다.
- 위 결과는 이번 범위의 실험과 검증 완료를 뜻한다. 모든 하드웨어의 전역 최적이나 모든 설정의 TPS 증가를 보장한다는 뜻은 아니다.
'''
report=HERE.parent/'REPORT.md'
s=report.read_text();marker='## 2026-10-14 tree 후속 최적화'
review_marker='<a id="precommit-review"></a>'
suffix='\n\n'+s[s.index(review_marker):] if review_marker in s else ''
s=s[:s.index(marker)].rstrip()+'\n'+section+suffix
report.write_text(s)
print('Updated',report)
