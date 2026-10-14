"""Record the user's measurement rules and explain the execution layout."""
import json
from pathlib import Path
HERE=Path(__file__).parent
d=json.loads((HERE/'MEASUREMENT_REVIEW.json').read_text())
parts=[r'''
<a id="measurement-and-layout-review"></a>

#### 2026-10-13 추가 질의: 준비 비용·batch miss·LM head replica·packed 의미

이번 변경은 benchmark의 측정·집계 규칙이다. 기존 inference runtime 77개 파일은 frozen hash 그대로이며 기본 알고리즘/정책을 바꾸거나 새 full-model 성능 실험을 한 것이 아니다. 아래 batch 지표와 준비 비용은 보존된 원시 기록의 재집계다. 이 변경은 미커밋이며 origin/MLSys-ver0.1@f12d0f1에는 포함되지 않는다.

##### 1. Graph 준비를 얼마나 제외하는가?

이전의23.77%/22.07%는 준비가 겹친 **step 전체 시간**의 비율이다. 준비 label의 CPU 시작·끝을 실제 decode step과 교차시키고 두 GPU/중첩 label의 겹침을 합집합으로 계산한 값은 다음과 같다. 최초 엔진 초기화·예열까지 포함한 모든 graph 준비 비용은 아니다.

| 첫480 순회 | Full-B 집계 시간 | 그 안의 관측된 준비 구간 합집합 | 비중 | 준비가 겹친 step 전체 시간 |
|---|---:|---:|---:|---:|
''']
for arm in ('wait','ar'):
    c=d['cells'][arm]['measured_first']['preparation_full_batch']
    parts.append(f"| {arm} | {c['seconds']:.3f}s | {c['interval_union_seconds']:.3f}s | {100*c['interval_union_fraction']:.2f}% | {c['affected_step_seconds']:.3f}s |\n")
parts.append('\n부분 batch·출력 경계를 포함하는 전체 decode 시간에서 관측된 준비 구간의 비중: ')
parts.append('; '.join(f"{arm} {100*d['cells'][arm]['measured_first']['preparation_all_decode']['interval_union_fraction']:.2f}%" for arm in ('wait','ar'))+'.\n')
parts.append(r'''
**사용자 요청을 반영한 새 주 집계 규칙:** full-B/non-cap/non-clip 조건에 더해 관측된 graph 준비와 겹친 step을 제외한다. 제외 시 시간·반환 token·AL·hit를 함께 제외한다. 준비 시간만 분모에서 빼고 그동안 반환한 token을 남기지 않는다. 원시 시간과 기존 summary는 보존한다. 시간 임계값으로 느린 step을 지우는 방식은 사용하지 않는다.

`ssd/bench/mlsys_measurement.py`를 native/sensitivity/prewarm/repeat harness 네 개에 연결했다. 새 실행은 target/draft의 capture label 외에도 CUDA graph context, input/root/proxy/accept constructor, 새 draft forest/attention plan 준비를 기록한다. 이전 trace에는 추가 계측이 없으므로 과거 구간을 완벽히 복원했다고 주장하지 않는다. Runtime 내장 TPS 로그는 기존 정의이고 **benchmark JSON의 summary가 새 정의**다. `summary_with_preparation`, 원시 step·capture 기록은 함께 보존한다. CUDA-event profile run은 TPS 근거로 사용하지 않는다.

두 번째 순회의 full-B 대상 step에는 기록된 준비 구간이 없어서 TPS292.94→290.72 결과는 그대로다. 그러나 전체 decode에는 Wait6.521s/AR1.708s의 준비 구간이 남았다(주로 부분 batch). 따라서 모든 inference에서 준비가 사라졌다는 뜻은 아니다. 부분 batch를 집계에서 제외하더라도 실제 서비스에서는 해당 shape도 준비해야 한다.

**Chain과 tree 모두 준비를 inference 이전으로 옮길 수 있다.** Chain은 설정된B·K1/K2 등 비교적 적은 shape를 초기화 때 준비하기 쉽다. Shape/backend/설정이 바뀌는 모든 chain까지 자동으로 준비됐다고 할 수는 없다. Tree도 모든 topology를 열거할 필요는 없다. 최대B·노드수·context/page 범위·phase round 폭을 정하고 capacity별 graph를 준비하면 topology와 token은 실행 중 tensor 값으로 바꿀 수 있다.

완전 준비의 범위는 target pre/post,proxy/accept,draft glue,P1/P2 arena·input/root graph,miss JIT의 batch shape 모두다. 기존63개 graph·약27~30초 실험은 **일부 packed target 경로의 사전 준비**였고 전체 draft/수락까지 준비한 결과가 아니다. 전 범위 준비는 startup/메모리 비용과 padding의 절충이 있다. 설정한 운영 범위에서 측정 전 graph/plan key 수를 고정하고, 검증용 전체 입력 순회 중 새 key/capture가0인지 확인하는 것을 완료 조건으로 둔다. 이 전체 prewarm 구현/검증은 아직 완료되지 않았다.

##### 2. Batch hit와 실제 fallback을 별도로 보고한다

`m`을 한 step에서 cache miss인 요청 수라 하면 request hit 평균과 함께 `P(m=0)`, `P(m>0)`, `P(0<m<B)`, `P(m=B)`, 실제JIT step/request 비율, AR로 처리된 miss 비율, m의 histogram을 기록한다. 실제JIT는 이 구현에서 miss이며 `valid_k>0`인 응답으로 판별하고, miss이며 `valid_k=0`이면 AR이다. Draft forward 호출 횟수와 같은 지표는 아니다.

같은 Wait 기록에 AR를 적용했다고 가정하는 값은 별도 이름으로 보존한다. 미래 문맥·cache 상태가 바뀌지 않는다는 가정의 분류 값이며 실제 AR 실행 성능의 예측값은 아니다. Request들이 독립이고 같은 hit 확률h를 가질 때만 all-hit=h^B이다. 실제 분석은 독립성을 가정하지 않고 기록된 joint histogram을 사용한다.

두 번째480 순회, 동일한 full-B/non-cap/non-clip/non-preparation 집계:

| 지표 | Wait | Mixed-miss AR |
|---|---:|---:|
''')
for label,key in [('Request hit','request_hit'),('모두 hit인 step','all_hit'),('하나라도 miss인 step','any_miss'),('Hit/miss 혼합 step','mixed'),('모두 miss인 step','all_miss'),('실제 JIT를 수행한 step','actual_jit_step'),('JIT proposal을 받은 요청 비율','actual_jit_request'),('Miss를 AR로 처리한 요청 비율','miss_ar_request')]:
    parts.append(f"| {label} | {100*d['cells']['wait']['measured_second']['batch_metrics'][key]:.2f}% | {100*d['cells']['ar']['measured_second']['batch_metrics'][key]:.2f}% |\n")
parts.append(r'''
이 full-B 표의 all-miss는 모두 각 고정8요청 group의 첫 step이었다. Group 첫 step을 제외한 별도 표에서는 Wait JIT step89.24%, AR0%다. 실제 serving에서 all-miss가 절대 생기지 않는다는 뜻은 아니다. 새 admission/작업 부하에서 다시 측정해야 한다. Request별 cache 구성은 독립성을 유지하지만 batch 크기·정책 변경 후의 관측 hit율까지 동일하다고 가정하지 않는다.

##### 3. K1/K2와 파라미터 선택 순서

K1/K2는 draft phase의 순차 forward round 수다. 각 round는 여러 요청·위치·root·분기를 batch로 처리하므로 K1=4가 draft가 총4token만 계산한다는 뜻은 아니다. 일반 chain의 proposal1개에서도 수락하면 bonus token까지 반환할 수 있어 기대 AL=1+alpha이며 AR의항상1과 다르다. 다만 draft 준비비용을 보상하지 못할 수 있다.

현K4/2가 최적이라는 근거는 없다. 순수 실행 최적화는 같은 입력·작업의 latency로 먼저 판별할 수 있지만 AL/hit/TPS를 바꾸는 정책은 합리적인 파라미터 영역을 먼저 정한 뒤 판단한다. 순서는 준비 비용 분리 → P1 완료/Proxy 도착 및 P2 완료/다음 target 요청 deadline 확인 → K1/K2/exit/tree 예산의 좁은 탐색 → 고정 설정에서 Wait/AR 대조 → 각 방법의 튜닝된 최종점 비교다.

현재K2는 P2 round와 miss JIT 깊이를 함께 정한다. 후속 진단에서는 P2 설정을 고정하고 miss JIT 깊이만 바꾸는 대조도 필요하다. 추가 공개 tuning knob의 채택과는 별개인 인과 분석이다. Miss latency가 클수록 AR의 시간 이득이 커질 수 있지만 포기한 AL도 같이 커질 수 있어 두 항을 측정해야 한다. 특정 miss-count threshold는 아직 채택하지 않았다.

##### 4. Full LM head replica의 효과를 TP1 stream 실험으로 대신 판단하지 않는다

사용자가 말한 기법은 논문4.4절의 **rank0 full-vocabulary LM head 복제 + exit 시 TP gather 제거 + 별도 stream**이다. 논문5.1절은 dense70B/TinyLlama,RTX PRO6000 Blackwell3장,targetTP2이고,5.5절은 해당 구성에서 노출된 proxy 계산이 hit step의1.7%라고 보고한다. 이 숫자는 적용된 구성의 breakdown이며 ON/OFF로 분리한 향상률은 아니다.

최근7B/8B 검사는 targetTP1이다. 이미 full head가 한 GPU에 있으므로 TP gather 제거 효과를 측정할 수 없다. 최근 B1의+0.3~1.5%와B8의−2.6~7.3%는 이 TP1 조건에서 stream을 추가한 효과이며 Blackwell/TP2 replica를 제거할 근거가 아니다. 과거3090/AWQ70B/TP4 replica probe는 중립에 가까웠지만 그것 역시 논문Blackwell 결과를 대신하지 않는다. 관련 기록: `PSD/ssd/experiments/proxy_async_overlap/final_rematch/RESULTS.md` 및 실행script의sm_86/AWQ/TP4 설정.

같은 총 연산량이어도 동시 실행은 유휴 자원을 채워 빨라질 수 있고, SM/register/cache/bandwidth 경쟁·stream/event 비용으로 느려질 수도 있다. “이론상 같거나 무조건 빠르다”는 보장은 없다. TP에서는 rank0의 과부하가 다음 layer collective에서 다른 rank를 기다리게 할 가능성도 봐야 한다. Blackwell에서 여유가 더 컸다는 것은 가능한 설명이며 아직 비교 실험으로 분리한 원인은 아니다. [NVIDIA 비동기 실행 문서](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

필요한 검증은 같은 모델·장비에서 B1/B>1 각각 (a) sharded head+gather, (b) full replica+같은 stream, (c) full replica+별도 stream의 대조다. Proxy 완료·target 후반 layer·전체 target·P2 시작/완료와 TP 대기를 함께 본다. 이 factorial 검증은 이번에 새로 실행하지 않았다.

##### 5. Packed란 무엇을 바꾸는가?

예: B4에서 요청A만 proposal4개, B/C/D는0개. 각 요청의 마지막 확정 token r도 입력하므로 실제 query 수는5,1,1,1이다.

```
Padded (20행):
A [rA a1 a2 a3 a4]
B [rB  _  _  _  _]
C [rC  _  _  _  _]
D [rD  _  _  _  _]

Packed (8행):
[rA a1 a2 a3 a4 | rB | rC | rD]
요청 경계 offset = [0,5,6,7,8]
```

네 요청을 한 문장으로 합치는 것이 아니다. 요청별prefix/page table/경계를 유지한 채 쓸모없는query행을 빼고 한 번의 batch 계산에 넣는다. Chain이면 각 요청 안의 causal mask를 쓰고, tree이면 노드가 자기 조상만 보는 mask를 쓴다. Tree A가 `rA→a→c`, `rA→b→d`이면 저장 순서를 `[rA,a,b,c,d]`로 펴도 c는b/d를 보지 않는다. Mask가 조상 관계를 보존한다. 실제구현은8query 단위 정렬을 사용하므로 모든 shape에서 padding이 완전히0은 아니다.

현재 fixed packed는 transformer/LM head 입력을 줄이지만 결과를 다시 `[B,N+1,V]`로 펼친다. 위 예에서8행×V를20행×V로 복원한다. Dummy행을 mask로 무효화하므로 실제 후보가 추가되는 것은 아니지만 proxy softmax/residual 점수·수락 준비가 그 padded 폭으로 계산된다. **Proxy·수락까지 padding 제거**는 원래8행과요청별offset을 그 단계까지 유지하자는 뜻이다. 선택 수식이나 proposal 법칙을 바꾸자는 제안이 아니다. 아직 구현하지 않았다.

수치 차이는 packing으로 GEMM/attention shape와축약 순서가 달라졌을 때의 target logits 차이다. 기존Llama3 half-short에서 argmax46/48만 같았다. Mask bug와 허용 가능한 부동소수점 차이를 구분하는 추가 검사 없이 무조건 정상/완전동일이라고 하지 않는다.

##### 6. Query 감소와 latency, 모델 차이

Half-short B8에서 query72→48의 동일 입력 비교(4090). 각값은 별도 CUDA interval 측정의 median이며 wall시간에 기계적으로 더하지 않는다.

| 모델 | 구간 | Padded → Packed |
|---|---|---|
''')
for model,cells in d['packed_components'].items():
    a=next(c for c in cells if c['shape']=='half_short' and c['implementation']=='padded')
    b=next(c for c in cells if c['shape']=='half_short' and c['implementation']=='fixed')
    for label,key in [('Target pre','batch_target_pre'),('Target post','batch_target_post'),('Proxy','batch_target_proxy')]:
        parts.append(f"| {model} | {label} | {a['cuda_components_ms'][key]:.3f}→{b['cuda_components_ms'][key]:.3f}ms |\n")
    parts.append(f"| {model} | Target wall | {a['wall_ms']:.3f}→{b['wall_ms']:.3f}ms |\n")
parts.append(r'''
Llama2는transformer구간 자체가 줄지 않았고,0.12ms/0.62% 증가는 작은 단일실행 차이이므로 안정적인 악화라고 단정하지 않는다. Llama3는pre/post가 줄었고proxy는 거의 그대로다. 따라서 이번 Llama3 이득을 “큰 vocabulary의 proxy가 빨라졌기 때문”으로 설명하면 맞지 않는다.

두 모델은 V32000/128256,KVheads32/8,FFN11008/14336,FP16/BF16으로 실제계산형태가 다르다(hidden4096과Qheads32는같음). Query 감소로 줄지 않는 weight읽기·launch비용, GEMM타일/점유율, 요청당query를묶던attention을query별sequence로펼친 비용이 상쇄할 수 있다. **확인된 구간 차이**와 **아직검증하지 않은 kernel병목 가설**을 구분한다. 특정 모델/shape의 kernel 선택이 덜 적합할 가능성은 있으며, 정확한 원인별기여도는 추가 profiling/동일attention-layout대조가 필요하다.

##### 7. “남은 CPU/phase 시간”과 “transformer 자체 가속 아님”의 의미

P1/P2 total에는 이미 root입력 준비·forward·tree처리·metadata등 많은 작업이 포함된다. 남은CPU처리를 별도 비용으로 또 더하자는 뜻이 아니다. 총시간을 더 줄이려면 어떤 내부처리가 지배적인지 분해해야 하며, phase밖의요청/응답준비도 확인해야 한다. 또한P1이35→30ms로줄어도proxy가20ms에오면여전히늦으므로 phase완료시각을사용가능window와비교해야한다.

“Transformer 자체 가속으로 주장하지 않음”은 추가 결함이나 승인조건이 아니다. 이식한불필요연산제거는proxy/accept를빠르게했고layer본체는거의같았다는**개선위치의표기**다. 해당구간 latency가줄고정확성을보존하면그대로유효한최적화다. Packed는별도변경이며위Llama3에서는layer본체도줄었다.

검증: 중첩 준비 구간의 합집합, 이전 step에서 시작된 준비, 시간·token·AL·hit의 동일 모집단, miss와 실제 JIT의 구분, 초기 step 분리, raw summary 보존을 6개 CPU 테스트로 확인했다. GPU 4에서 계측을 설치한 CUDA graph가 정상 capture/replay되고 기록을 읽을 수 있는지도 확인했다. 새 측정 코드로 전체 480개 입력을 다시 실행한 결과는 아니다. 계측은 향후 실험에 사용하며 기존 원시 기록과 archive는 보존했다.

재집계: `measurement_review.py --raw-root <기존 raw root>` → `write_measurement_review.py`. 근거는 `MEASUREMENT_REVIEW.json`. 이번 측정 코드와 문서는 `MEASUREMENT_SOURCE.tar.gz`와 `MEASUREMENT_ARTIFACTS.json`에 별도로 보존한다. 후속 순서는 전체 shape의 사전 준비 검증, 합리적인 phase 예산 탐색, packed 후속 처리 개선, 같은 조건의 mixed-miss 및 TP replica 분리 검증이다.
''')
report=HERE.parent/'REPORT.md';s=report.read_text();marker='<a id="measurement-and-layout-review"></a>'
if marker in s:s=s[:s.index(marker)].rstrip()+'\n'
report.write_text(s+''.join(parts))
print('Updated',report)
