"""Update only the final extension review section in the single handover report."""
import json
from pathlib import Path

HERE=Path(__file__).parent
d=json.loads((HERE/'FINAL_ANALYSIS.json').read_text())
parts=[r'''
<a id="extensions-final-review"></a>

#### 최종 검토 — 적용 상태와 직접 latency 근거

**승인한 새 모델 지원·B>1 tree 실행·연산 최적화는 이식했다. Packed tree·mixed-miss·proxy side stream은 검토용으로 구현·검증했지만 기본 활성화하지 않았다.** 아래 성능 수치는 새 B>1 실행기 내부의 최적화 전후다. 원본 논문 B1 대비 개선율로 인용하면 안 된다. 논문 B1 경로는 유지했고 추가 기능 회귀도 대조했다.

| 항목 | 기존 동작 → 이번 동작 | 적용 상태 |
|---|---|---|
| 새 모델 | AMD config 재생성 뒤 dtype 불일치 → 실행 dtype 일치. 모델 family/크기만 검사 → tokenizer ID mapping 검사. Qwama 넓은 P1 → attention workspace 확보 | Native 이식. 두 모델 pair 및 SSD smoke 통과 |
| B>1 tree | 기존 B1 tree 경로 → 요청별 tree/cache/epoch/예산을 보존하면서 draft round를 묶어 실행, target tree attention·수락·KV commit 연결 | B>1 기본 경로. B1 간소화 CPU 수락기는 유지 |
| Draft 연산 결합 | 작은 mask/fanout/slot 처리를 각각 실행 → batched staging executor에서 결합 kernel 사용 | B>1 기본 활성화. B1의 기존 연산 선택 유지 |
| Draft metadata 일괄 처리 | root/executor별 CPU readback → 결과를 묶어서 전송·파싱 | B>1 기본 활성화 |
| Target 불필요 계산 제거 | 사용하지 않는 마지막 residual 갱신까지 수행 → 필요한 ladder만 계산. T>0 계약에서 argmax/one-hot 준비 생략 | 기본 활성화. 난수·수락분포 검사 |
| Packed tree | 최대 길이 padding → 실제 query를 모으고 query당 고정 attention plan 재사용. 제거할 padding이 없으면 원래 경로 사용 | **검토용**. Shape별 이득이 달라 일반 기본값 보류 |
| Mixed-miss tree | 하나라도 miss면 해당 JIT draft 완료까지 대기 → 일부 hit가 있을 때 miss 행만 proposal 0개로 target AR 처리 | **검토용**. 현재 시험 정책은 `0<miss<B` 전부 AR. 이 threshold를 채택했다는 뜻 아님 |
| B>1 proxy side stream | Proxy projection/sampling 후 후반 layer 실행 → 별도 stream에서 겹쳐 실행 | **보류 권고**. 동일 target 입력에서 이득 확인 못함 |

새 public 탐색 옵션을 여러 개 채택하지 않는다. 기본값은 **padded tree + 연산 결합 + bulk metadata + 불필요 계산 제거** 한 조합이다. Packed/mixed 선택은 `ssd/bench/mlsys_tree_candidates.py`의 비교용 adapter에 있고 runtime 기본 동작은 아니다. Miss draft의 DS/P1형 변경, root/위치/tree AL 점수 변경은 이번에 이식하지 않았다.

##### 1. 측정 설정과 비교 범위

- 모델: LayerSkip-Llama2-7B + AMD-Llama-135m, LayerSkip-Llama3-8B + Qwama-0.5B-Instruct. **Dense full model, target TP1 + 별도 draft GPU**. 70B/AWQ/TP4 검증을 완료했다는 뜻은 아니다.
- 주 설정: B=8, target/draft T=0.7, K1=4, K2=2, early-exit **index21(22번째 layer)**, P1 generation/verification=8/8, P2=4/4, sibling C=3, roots/position=2, draft fanout=2, async fanout=3. Root residual·기존 관측 위치 점수·tree top-m 정책 유지.
- 온라인: 저장된480입력 전체(MT-Bench/translation/summarization/QA/math/RAG 각80), 최대 입력512·출력64, warm48, target seed11013·draft seed231. 이는 해당 저장 집합 전체이며 원본 여섯 benchmark의 모든 예제를 뜻하지 않는다.
- 프로파일: 같은 집합의 균등 추출48입력, warm48. Graph capture step·부분 batch·cap/clip을 제외하고 **같은 group/step**만 대조. 같은 실행의 중첩 구간을 단순 합산해 total로 제시하지 않는다. Profile TPS는 사용하지 않는다.
- 같은 GPU pair 내에서 비교. `orion` RTX4090과 `nova` RTX3090 절대값을 합치지 않는다. GPU 프로세스/온도/clock telemetry를 보존했다. 최종 표에 쓰인 campaign에 기록된 외부 GPU PID는 없다.
- NUMA: orion6/7 및4/5는 NUMA1. nova는2–7 모두 NUMA0(0–15,32–47). 주 `profiles_v2` 및 Llama3 sensitivity는 nova2/3 + local CPU affinity. 보조 ablation·prewarm은 같은 pair/CPUset으로 대조했지만 nova에 원격 NUMA CPUset을 사용했다. 따라서 그것을 최적 CPU 배치의 절대 latency로 인용하지 않는다.
- 요청한 집계 정의대로 활성 B가 지정 B와 같은 step 중 cap/clip step을 제외한다. 분자와 분모를 함께 제외하며, 전체 반환 token/전체 decode 시간(raw TPS)도 보존한다. EOS/tail 분포가 변하면 남는 측정 모집단도 변할 수 있다.

##### 2. Target과 draft P1/P2를 직접 분리한 결과

주 근거는 nova3090의 같은 출력·같은 수락 event 대조다. 아래는 CUDA interval 평균(ms), capture 제외 후 양 arm 공통 step(Llama2 128, Llama3 45)이다. P1/P2 replay는 forward와 해당 tree round의 GPU 처리이며 total에는 metadata 등이 포함된다.

| 직접 측정 구간 | Llama2 전 → 후 (ms) | 변화 | Llama3 전 → 후 (ms) | 변화 |
|---|---:|---:|---:|---:|
''']
for label,key in [('Draft P1 replay','draft/batch_p1_replay'),('Draft P1 total','draft/batch_p1_total'),
                  ('Draft P2 replay','draft/batch_p2_replay'),('Draft P2 total','draft/batch_p2_total'),
                  ('Target exit 이전','target/batch_target_pre'),('Target exit 이후','target/batch_target_post'),
                  ('Target proxy 계산','target/batch_target_proxy'),('Target 수락 처리','target/batch_tree_accept')]:
    a,b=[d['profiles'][m][key] for m in ('llama2','llama3')]
    parts.append(f"| {label} | {a['before_ms']:.3f} → {a['after_ms']:.3f} | {a['percent']:+.2f}% | {b['before_ms']:.3f} → {b['after_ms']:.3f} | {b['percent']:+.2f}% |\n")
parts.append(r'''
![직접 latency](extensions/figs/01_component_latency.png)

**Target transformer 자체가 빨라졌다는 결과는 아니다.** Draft tree round와 target proxy/수락 처리의 비용 감소다. 빨라진 P1이 proxy 도착 전에 끝나면 proxy 대기 시간이 늘 수 있다. 이는 일을 빨리 끝내고 기다리는 것이므로 P1 최적화의 역효과로 읽으면 안 된다.

각 최적화도 따로 제거해 확인했다(`FINAL_ANALYSIS.json`의 `ablations`):

| 한 번에 바꾼 항목 | Llama2 직접 latency | Llama3 직접 latency |
|---|---|---|
''')
for label,edge,keys in [('Draft 결합 kernel','reference->fused',['draft/batch_p1_replay','draft/batch_p2_replay']),
                      ('Bulk metadata','fused->bulk',['draft/batch_p1_total','draft/batch_p2_total']),
                      ('Target ladder 마지막 불필요 갱신 제거','bulk->trim',['target/batch_target_proxy','target/batch_tree_accept'])]:
    vals=[]
    for m in ('llama2','llama3'):
        c=d['ablations'][m][edge]; vals.append(' / '.join(f"{c[k]['before_ms']:.3f}→{c[k]['after_ms']:.3f} ms" for k in keys))
    parts.append(f"| {label} | {vals[0]} | {vals[1]} |\n")
parts.append(r'''
표의 첫 두 행은 P1/P2 순서, 마지막은 proxy/수락 순서다. Bulk 처리의 replay 자체는 거의 같고 metadata를 포함한 total이 줄었다. 이 보조 ablation에서도 출력·수락 event가 일치했다. P1/P2 metadata 경계와 overlap이 있으므로 total 감소량을 그대로 노출된 step 절약 시간으로 더하지 않는다.

T>0에서 greedy 계산을 생략하는 부분은 별도 동일 tensor·동일 graph 입력40회 교대 측정으로 확인했다. 4090 B8 수락 kernel은 vocab32k에서 trim 이후0.563→0.531ms, vocab128256에서2.788→2.626ms. B2에서도 감소했다. 같은 seed10개에서 경로/recovery 일치. 이 microbenchmark는 CPU affinity를 고정하지 않았으므로 CUDA interval을 주 근거로 사용한다.

##### 3. Packed tree: 어디서 이득이 생겼나

요청마다 실제 노드 수가 달라도 query당 고정 attention sequence를 사용해 **초기 packed 구현의 매 step FlashInfer plan 재생성을 제거**했다. 비교 기준인 padded tree도 이미 shape별 plan을 재사용하므로, 이는 padded 대비 추가 절약이라고 해석하면 안 된다. 같은 query 수 bucket/page bucket이면 tree topology는 tensor 내용으로 바꾸며 graph를 재사용한다. Query 수가 줄지 않는 shape는 원래 padded 실행기로 돌아간다.

실제 모델과 길이128 prefilled KV에 같은 synthetic token/tree를 넣은 target-only 검사다. Draft/AL/hit을 바꾸지 않고 target 호출 wall latency를 비교했다. 아래 B8,20회 median. 토큰 값/shape는 인위적이며 자연 입력의 발생 빈도를 뜻하지 않는다.

| 모델 / 4090 | Tree shape | Padded | 매번 plan 갱신 | 고정 plan | 고정 plan 변화 |
|---|---|---:|---:|---:|---:|
''')
for m in ('llama2','llama3'):
    cells=d['fixed_target'][m]['cells']
    for shape,label in [('uniform','8개씩'),('half_short','8개×4행 + 2개×4행'),('one_hit','8개×1행 + 0개×7행')]:
        vals=[next(c['wall_ms'] for c in cells if c['batch']==8 and c['shape']==shape and c['implementation']==name and not c['stream']) for name in ('padded','replan','fixed')]
        parts.append(f"| {m} | {label} | {vals[0]:.3f} ms | {vals[1]:.3f} ms | {vals[2]:.3f} ms | {(vals[2]/vals[0]-1)*100:+.2f}% |\n")
parts.append(r'''
![Packed shape별 비교](extensions/figs/02_packed_shapes.png)

일부 행만 길 때 이득은 확인됐지만 Llama2의 절반-short shape에서는 이득이 없었다. 따라서 **query 수 감소만 보고 모든 shape에서 빠르다고 가정하지 않는다.** One-hit 예시는 mixed-miss AR에서 만들 수 있는 극단 shape이며 현재 workload가 대부분 그런 shape라는 뜻은 아니다.

첫 graph capture도 별도로 검토했다. Packed lazy/prewarm full480 대조에서 출력/AL/hit/event는 같았다:

| 모델 / 3090 | Verification interval lazy → prewarm | 별도 startup 비용 | 준비 graph |
|---|---:|---:|---:|
''')
for m,c in d['prewarm'].items():
    parts.append(f"| {m} | {c['lazy']['verify_ms']:.3f} → {c['prewarm']['verify_ms']:.3f} ms | {c['startup_seconds']:.2f} s | {c['graphs']} |\n")
parts.append(r'''
이는 capture 비용을 서비스 이전으로 옮긴 효과이며 동일 GPU kernel의 steady replay가 그만큼 빨라졌다는 뜻이 아니다. 입력512/출력64·현재 node 예산에 맞춘 준비 범위다. 일반 max context/B/예산 전체로 확대하면 startup과 memory가 더 늘 수 있다.

수치 주의: packed는 GEMM/attention shape가 바뀐다. Fixed packed Llama3 B8 half-short에서 최대 logit 차0.21875, argmax 일치46/48이었다. One-hit는16/16, uniform fallback은72/72. Llama2 fixed B8 세 shape는 argmax 일치지만 bitwise logits는 다를 수 있다. Independent dense attention 대조는 통과했으나 **원래 padded 출력과 bitwise 동일한 최적화로 분류하지 않는다.** Greedy는 실행한 target logits의 argmax를 따르는 계약이다. 분포/AL이 바뀐 온라인 packed TPS로 실행 속도만의 인과를 주장하지 않은 이유다.

현재 결론: 고정 plan 구현은 유효한 후보이나 모든 shape에 기본 활성화할 근거는 부족하다. 목표 모델/GPU의 실제 query 분포에서 이득이 확인되는 범위를 검토한 뒤 채택한다.

##### 4. Proxy side stream: 이 서버에서는 기본값 채택 근거 없음

![Proxy stream 비교](extensions/figs/03_proxy_stream.png)

같은 target 입력의 B8 uniform에서 Llama2 19.190→19.312ms, Llama3 24.387→24.470ms로 짧아지지 않았다. Llama3 proxy 구간3.981ms가 별도 stream에서는7.435ms, 후반 transformer5.956→9.957ms가 됐다. 동시 실행 시 각 구간이 느려져 overlap 이득이 상쇄되는 관측이다. GPU 자원 경쟁과 일치하지만 hardware counter로 원인을 분리한 것은 아니다. 다른 stream 구간을 합산한 값은 total이 아니며, 위 전체 target wall 시간을 기준으로 판단했다. B2도 원시 자료에 있다. B1에 이미 있던 overlap의 필요성을 부정하거나 새 contribution으로 제시하는 결과는 아니다.

##### 5. Mixed-miss: 파라미터가 최적이 아닐 때의 판단

**2026-10-13 재감사 정정:** 아래 첫 실행 TPS에는 warm48에서 보지 못한 shape의 graph 생성 비용이 포함돼 있다. 따라서 replay 상태에서 mixed-miss가 이득이 없다는 결론의 근거로 쓰지 않는다. 아래 수치는 원시 관측으로 보존하고, 문서 끝의 `extensions-followup-review`에서 capture 추적과 전체 입력 반복 결과를 함께 확인한다.

**판단은 (1) 같은 파라미터의 wait/AR TPS·AL, (2) 인접 파라미터에서 방향 유지 여부, (3) 같은 상태의 AL 손실 대비 필요한 시간 절약으로 나눈다.** 다른 파라미터로 얻은 TPS 한 점을 비교하지 않는다.

이번 sensitivity는 packed를 끈 동일 기본 tree에서 K2=1/2/3 각각 full480 wait/AR 한 쌍이다. K1=4·exit index21·tree budget은 고정했다. `duet_jit_short=True` 기본 동작 때문에 **K2는 P2 횟수와 miss chain 깊이를 함께 바꾼다.** P2만의 독립 효과가 아니라 실제 설정 전체에서 policy가 유리한지 보는 검사다. 한 seed/설정당 한 실행이며 전역 최적 parameter sweep이 아니다.

| 모델 / GPU | K2 | B | Wait → AR AL | Wait → AR hit | Wait → AR TPS* | TPS 변화 / 95% CI |
|---|---:|---:|---:|---:|---:|---|
''')
for m,cs in d['sensitivity'].items():
    for k,c in cs.items():
        w,v,co=c['wait'],c['ar'],c['comparison'];ci=co['tps_percent_ci']
        parts.append(f"| {m} / {'4090' if m=='llama2' else '3090'} | {k} | 8 | {w['al']:.3f}→{v['al']:.3f} | {w['hit']*100:.2f}→{v['hit']*100:.2f}% | {w['tps']:.2f}→{v['tps']:.2f} | {co['tps_percent']:+.2f}% / [{ci[0]:+.2f}, {ci[1]:+.2f}]% |\n")
parts.append(r'''
TPS*는 앞에서 정의한 full-B/non-cap/clip 모집단이다. CI는 같은480입력의60개 고정8-request group을 짝지어10,000회 bootstrap한 값이다. GPU/seed 반복의 불확실성까지 포함하는 CI가 아니다. AL은 이번 step의 확정 root/recovery를 포함한다. 초기에 모든 행이 miss인 step은 여전히 JIT를 수행하므로 전체 miss 평균 AL이1이어야 하는 것은 아니다.

![Mixed-miss TPS와 AL](extensions/figs/04_mixed_miss.png)

전체 decode 시간 기준 raw TPS도 별도로 보존했다. 예를 들어 Llama2 K2=1은 filtered −2.42%지만 raw326.41→333.52 TPS다. Llama2 K2=3은 filtered 감소지만 raw350.19→351.35다. **끝부분의 부분 batch를 포함하느냐에 따라 방향도 달라질 수 있으므로**, 고정 B의 정책 효과와 요청 전체 처리율을 구분한다. 각 cell의 raw/retention은 `FINAL_ANALYSIS.json` 및 `TABLES.txt`에 있다. 이 결과로 최적 튜닝 후 mixed-miss가 반드시 나쁘다거나 좋다고 단정하지 않는다.

좀 더 직접적인 한-step 기준은 다음과 같다. Wait의 기대 반환량/시간을 `A_W,T_W`, AR의 값을 `A_A,T_A`라 하면

\[
\text{AR가 유리}\iff\frac{A_A}{T_A}>\frac{A_W}{T_W}
\iff \underbrace{T_W-T_A}_{\text{실제 절약 시간}}
>T_W\frac{A_W-A_A}{A_W}.
\]

Miss m개, miss당 wait 기대 AL을 g, hit 행 기대 AL 합을 H라고 단순화하면 `A_W=H+mg`, `A_A=H+m`이다. 따라서 필요한 절약 시간은 **`T_W·m(g−1)/(H+mg)`**다. Miss 수만 같아도 g·hit tree 깊이·target/draft 시간에 따라 기준이 달라진다. 하나의 보편적인 miss-count threshold가 이론적으로 결정되는 구조가 아니다.

기존 verifier alpha에서 구한 조건부 기대 AL과 **진단 hook이 없는 원래 profile의 시간**을 대응시킨 사후 검사:

| 모델 | Miss 개수 | 대응 step 수 | 포기하는 기대 token 합 | 예상 노출 JIT 절약 median | 필요한 절약 median | 조건 만족 step 비율 |
|---|---:|---:|---:|---:|---:|---:|
''')
for m,c in d['expected'].items():
    for miss in ('1','2','3'):
        x=c['miss_counts'][miss]
        parts.append(f"| {m} | {miss} | {x['steps']} | {x['expected_token_loss']:.3f} | {x['estimated_saved_ms']:.3f} ms | {x['break_even_saved_ms']:.3f} ms | {100*x['predicted_win_fraction']:.1f}% |\n")
parts.append(r'''
이 표는 K2=2·48입력 중 해당 조건만의 작은 사후 표다. 토큰 손실은 고정 proposal에 조건화한 기대값이고 시간 절약은 `batch_miss_draft − no-miss metadata` 근사다. 현재 padded canvas가 같은 경우를 가정하며 실제 AR 후의 CPU·target 변화/미래 cache 상태까지 측정한 oracle가 아니다. Target 분포를 미리 안다고 가정해 온라인 선택기에 넣지 않았다. 진단 run의 출력/수락 event는 일반 profile과 일치했지만 CPU readback 때문에 **진단 run TPS는 폐기**했다.

현재처럼 `0<miss<B`이면 전부 AR로 바꾸는 정책은 아직 기본값으로 확정하지 않는다. 아래 사후 근사는 graph 준비와 이전 step의 비동기 작업 대기까지 엄밀히 분리한 정책 검증이 아니다. Miss1도 위 조건 만족 비율이 약절반이며, miss가 더 많으면 AL 손실이 커진다. 후속 채택의 조건은 **최종 parameter에서 wait 대비 TPS 개선과 이 시간/AL 손익 관계가 함께 재현되는 것**이다. 학습 parameter나 임의 threshold를 이번 production에 추가하지 않았다.

##### 6. 기능 검증·제외한 자료·남은 한계

- Frozen v2 runtime의234회 targeted unittest 실행: failure0/error0/skip0. Tree ancestor attention을 independent dense reference에 대조, batched vs separate executor 비교, proposal/residual 수락 법칙, zero-node AR 분포30,000회, T0, 요청 상태·KV·wire·shape 검사를 포함한다. `nova_latest/mlsys_ver01/extensions/CHECKS.json`과 suite 로그 보존.
- Full480 base→optimized 두 모델에서 출력·수락 event·AL·hit 동일. 각 개별 ablation도 동일. 실행 최적화가 root/위치/tree 선택 정책을 바꾼 결과가 아니다.
- B1 추가 회귀: f12d0f1 원본 snapshot과 현재 native 구현, 같은24입력/출력상한96. 출력2209token 및1125개 수락 event의 기존 필드 일치. 새 `tree/batch_size/verify_width` metadata만 추가. 이전 arm은 dtype/resource 안전 adapter가 필요했고 현재 arm은 native이므로 이 대조는 기능 회귀 근거로 사용한다.
- 기존 common harness는 safe-shutdown을 source text로 치환해서 새 native 코드에 적용할 때 시작 전 실패했다. Runtime 오류가 아니며 해당 시도는 제외하고 native harness로 완료했다. 실패 로그도 `b1_regression/after.log`에 보존했다.
- 광범위 `test_*tree*` discovery는 기존 분석 test가 없는 `tools.duet_calibration.analyze_tree_outcomes`를 import하여1 error. 따라서 “저장소 전체 테스트 모두 통과”라고 하지 않는다. Runtime targeted suite는 위234회 통과다.
- 초기 local screen의 외부 PID 점유 run, ENOSPC로 결과 저장을 못한 run은 최종 성능 표에서 제외했다. 다른 사용자의 GPU 프로세스는 종료하지 않았다.
- 70B/TP4/AWQ, 큰 B·긴 context·다양한 tree 예산, 목표 serving arrival/여러 seed에 대한 새 성능 주장은 아직 없다. Prefix cache는 이식한 기능을 새로 끈 것이 아니라 양 arm에서 cold-prefix 측정 조건을 사용했다.
- 이번 모든 결과는 위 고정 설정에서의 implementation 검토다. 원본 논문 설정/튜닝된 SSD 대비 우위를 새로 입증한 실험으로 인용하지 않는다. SSD는 호환성 smoke이며 최적 SSD throughput 탐색을 했다는 뜻이 아니다.

##### 7. 재검토·다른 서버 인수인계

1. 이 문서의 적용 상태 → `extensions/PORT.json` → `extensions/FINAL_RUNTIME_HASHES.json` 순서로 범위/실제 코드를 확인한다. Base는 `MLSys-ver0.1@f12d0f1`, source port는 `feat/duet-mlsys-coverage@1d9ace6`; 그 worktree의 미커밋 연구는 가져오지 않았다.
2. `extensions/FINAL_SOURCE.tar.gz`에는 최종 runtime/bench/tests, `RUNTIME.patch`에는 f12d0f1 대비 runtime 변경을 둔다. `port_approved.py`는 최초 선택 이식 기록이므로 **최종 코드를 재현하려고 다시 실행하면 후속 최적화를 잃는다.** 최종 source/patch를 사용한다.
3. `extensions/RAW_RESULTS.tar.gz`와 `ARTIFACTS.json`으로 원시 JSON·profile·telemetry·로그의 출처/hash를 확인한다. 외부 원본 경로는 `/data/chokwans99/duet-ver01-extensions-20261013`, 원격 사본은 `/data2/chokwans99/PSD-MLSys-ver01-extensions/results`다.
4. Raw archive를 새 디렉토리에 풀고 `python results/mlsys_ver01/extensions/finalize.py --raw-root <그 디렉토리>`로 표/그림을 다시 생성할 수 있다. `analyze.py`는 raw step에서 TPS를 독립 재집계하고 기록값과 일치해야 통과한다. 모델 경로·CPU affinity·GPU pair는 실행 plan을 해당 서버에 맞게 바꾼다.
5. 기본값에 추가 승인할 후보는 아직 없다. Packed의 shape별 이득/수치 차이, mixed-miss의 AL/TPS tradeoff, stream의 자원 경쟁을 각각 검토한다. 다른 서버의 root/위치/AL selector 변경과 합칠 때 먼저 실행 경로 parity를 고정한 뒤 정책별 ablation을 해야 한다.

그림 전체: [REVIEW.pdf](extensions/figs/REVIEW.pdf). 숫자 전체: [FINAL_ANALYSIS.json](extensions/FINAL_ANALYSIS.json), [TABLES.txt](extensions/TABLES.txt). AWQ 파일은 삭제하지 않았다. 비밀번호는 어떤 artifact에도 저장하지 않았다.
''')
report=HERE.parent/'REPORT.md'
s=report.read_text();marker='<a id="extensions-final-review"></a>'
followup_marker='<a id="extensions-followup-review"></a>'
followup=s[s.index(followup_marker):] if followup_marker in s else ''
if marker in s:s=s[:s.index(marker)].rstrip()+'\n'
report.write_text(s+''.join(parts)+('\n'+followup if followup else ''))
print('Updated',report)
