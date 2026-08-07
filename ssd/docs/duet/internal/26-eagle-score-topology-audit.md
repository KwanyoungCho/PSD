# DUET P2 EAGLE식 topology 점수 감사 (2026-08-07)

## 1. 결론부터

`duet_tree_policy=eagle`의 **실행 구현**은 끝나 있다. CPU 참조 구현,
eager GPU 구현, 전체 P2 CUDA graph 실행기가 같은 규칙을 수행한다. 하지만
이것은 SGLang EAGLE-2를 그대로 복사한 것이 아니라, DUET의 10개 proxy root와
temperature>0 비복원 추출/검증 규약에 맞춘 **EAGLE식 전역 확장 정책**이다.
장기 실험에서 P2AL과 P2 기여가 chain보다 낮았기 때문에 현재 주력 정책으로는
채택되지 않았다.

이번 감사에서 실제 P2 113 step의 점수와 topology를 기록했다. 현재 점수식은
다음과 같다.

```text
자식 경로 점수 = proxy(root)
               × q(첫 자식)
               × q(둘째 자식 | 첫 자식)
               × ...
```

여기서 `q`는 draft 분포에서 그 토큰이 가진 원래 확률 질량이다. 세 토큰은
확률에 비례한 비복원 추출로 뽑지만, 점수에는 추출 뒤 재정규화한 값이 아니라
추출 전 확률을 쓴다.

관측 결과는 두 가지다.

1. confidence가 proxy를 이기는 경우도 분명히 있다.
2. proxy가 극단적으로 쏠린 step에서는 낮은 confidence 자식이 여러 lane을
   차지해, 다른 root의 매우 높은 confidence 자식을 탈락시키는 문제도
   분명히 있다.

따라서 "confidence가 무시된다"는 진단은 틀리지만, "극단적인 proxy 비율을
그대로 곱해서 과도하게 지배하는 step이 있다"는 진단은 맞다.

## 2. 감사 데이터와 재현성

- 실행: seed 42, 각 dataset 1 prompt, output 96 smoke
- 실제 P2 topology: 113개
- root: step당 10개
- 첫 draft 뒤 후보: root당 3개, 총 30개
- 다음 draft에 들어간 부모: 전역 상위 10개
- 기록: `experiments/proxy_async_overlap/tree_sweep/
  eagle_score_trace_20260807/smoke_topology.draft.jsonl`

로그에는 각 노드마다 다음 값이 있다.

- `piv`: target early-exit가 보낸 root proxy 점수
- `raw_q`: 그 부모의 draft 분포에서 자식 토큰의 확률
- `path_conf`: root부터 해당 노드까지 `raw_q`의 곱
- `score`: `piv × path_conf`
- `par`: 부모의 root-local node 번호
- `tok`: 실제 token id

첫 draft의 30개 후보를 `piv × raw_q`로 다시 정렬했을 때, 실행기가 실제로
다음 forward에 선택한 10개를 **113/113 step에서 정확히 재현**했다. 아래
예시는 추정 topology가 아니라 실제 실행기 출력이다.

같은 코드를 eslab17의 별도 worktree에서 seed 123으로 병렬 실행해 116개
topology를 더 얻었다. 이쪽도 stable tie-break를 포함해 **116/116**을 정확히
재현했다. 독립 로그는 `eagle_score_trace_eslab17_s123_20260807/`에 있다.

## 3. 예시 A — proxy가 지나치게 지배한 경우 (trace 85)

10개 root 중 root 0의 proxy 점수는 0.761이고, 10개 합에서 차지하는 비율은
98.1%였다.

```text
root 0  proxy=0.761253
├─ n0  tok=29871  q=0.990950  score=0.754363  [다음 forward 선택]
│  └─ n3 q=0.993462 score=0.749431
│     └─ n6 q=0.988726 score=0.740982
│        └─ n7 q=0.952745 score=0.705967
├─ n1  tok=395    q=0.006677  score=0.005083  [다음 forward 선택]
│  └─ n4 q=0.966004 score=0.004910
└─ n2  tok=313    q=0.001913  score=0.001456  [다음 forward 선택]
   └─ n5 q=0.996058 score=0.001451
```

root 0은 confidence가 0.67%, 0.19%밖에 안 되는 두 약한 자식까지 다음
forward에 보냈다. 반면 아래 root들은 높은 confidence 자식을 갖고도 모두
탈락해 깊이 1에서 멈췄다.

| root | proxy | 가장 좋은 자식 q | 현재 점수 `proxy×q` | 결과 |
|---:|---:|---:|---:|---|
| 3 | 0.000530 | 0.506756 | 0.000268 | 깊이 1에서 중단 |
| 4 | 0.000423 | 0.643658 | 0.000272 | 깊이 1에서 중단 |
| 9 | 0.000398 | 0.697483 | 0.000278 | 깊이 1에서 중단 |

root 9의 자식은 root 0의 두 번째 자식보다 confidence가 104배 높다. 그러나
proxy가 약 1,912배 작아서 최종 점수는 오히려 18.3배 작다. 이 step에서는
proxy가 명백히 과도하게 지배한다.

proxy에 제곱근을 적용하면 점수는 다음처럼 바뀐다.

```text
score = sqrt(proxy) × path_confidence

root 0의 q=0.006677 자식: 0.00583
root 9의 q=0.697483 자식: 0.01392
```

이 경우 root 9가 root 0의 약한 자식보다 먼저 선택된다. 실제 첫-round
후보에 이 식을 적용하면 root 0의 약한 두 자리가 빠지고 root 3/4/9의 강한
자식이 들어온다.

## 4. 예시 B — confidence가 proxy 순서를 뒤집은 경우 (trace 56)

proxy가 항상 결과를 결정하는 것은 아니다.

| root | proxy | 선택 후보 q | 최종 점수 | 결과 |
|---:|---:|---:|---:|---|
| 3 | 0.028118 | 0.055825 | 0.001570 | 깊이 1에서 중단 |
| 7 | 0.004219 | 0.439547 | 0.001854 | 깊이 4까지 확장 |

root 3의 proxy는 root 7보다 6.67배 높지만 confidence가 낮아서 탈락했다.
root 7의 실제 확장 경로는 다음과 같다.

```text
root 7  proxy=0.004219
└─ n1 q=0.439547 score=0.001854
   ├─ n3 q=0.856227 score=0.001588
   │  └─ n6 q=0.999994 score=0.001588
   │     └─ n7 q=0.825342 score=0.001310
   ├─ n4 q=0.143570 score=0.000266
   └─ n5 q=0.000039 score≈0
```

즉 현재 구현은 confidence 기반 동적 선택을 실제로 하고 있다. 단지 proxy
격차가 매우 큰 step에서는 confidence가 그 격차를 이기기 어렵다.

## 5. 예시 C — proxy가 균형적인 경우 (trace 3)

이 step의 가장 큰 proxy 비중은 14.4%였고 유효 root 수(엔트로피 기반)는
9.66개였다. proxy 범위가 좁기 때문에 선택은 대부분 confidence가 결정했다.

| root | proxy | 선택된 자식 q | 점수 | 결과 |
|---:|---:|---:|---:|---|
| 4 | 0.002288 | 0.996672 | 0.002281 | 확장 |
| 5 | 0.002149 | 0.999988 | 0.002149 | 확장 |
| 6 | 0.001782 | 0.737108 | 0.001314 | 확장 |
| 9 | 0.001225 | 0.625658 | 0.000766 | 확장 |
| 0 | 0.001029 | 0.477256 | 0.000491 | 중단 |

현재 식에서 마지막 선택 점수는 0.000581이어서 root 0은 근소하게
탈락했다. 제곱근 proxy를 쓰면 root 0이 들어오고 한 root의 두 번째 후보가
빠지는 정도다. 균형적인 step에서는 proxy 완화의 영향이 작다.

## 6. 전체 113 step 통계

proxy 값은 top-10 후보의 부분 질량이므로 합이 항상 1은 아니다. 아래
`top 비중`은 step 안에서 10개 값을 다시 정규화해 concentration만 본 값이다.

| 항목 | 결과 |
|---|---:|
| top proxy 비중 중앙값 | 38.5% |
| top proxy 비중 최소 / 최대 | 14.4% / 98.1% |
| 유효 root 수 중앙값 | 6.15 / 10 |
| 깊이 4까지 간 root | 917 / 1,130 (81.2%) |
| 깊이 1에서 멈춘 root | 213 / 1,130 (18.8%) |

eslab17 seed 123도 같은 양상이었다: top proxy 비중 중앙값 40.4%, 깊이 4
root 937/1,160(80.8%), 현재 점수에서 다음 forward에 들어간 서로 다른 root
8.08개, 제곱근 점수에서 8.76개였다. 두 서버를 합치면 229 step·2,290 root
중 1,854개(81.0%)가 깊이 4, 436개(19.0%)가 깊이 1이었다. 따라서 아래
현상은 한 서버·한 seed의 우연으로 보이지 않는다.

현재 `Nv=8, C=3, F=4`의 예약 규칙 때문에 관측된 root view 크기는 3 또는
8뿐이었다. 즉 어떤 root는 `[3]`에서 멈추고, 살아남은 root는 사실상
`[3,3,1,1]`까지 간다. branch의 **내용과 살아남는 root는 동적**이지만,
root별 자원 배분은 상당히 거친 all-or-nothing 구조다.

첫 draft 뒤의 30개 후보에만 proxy 지수 `alpha`를 바꿔 오프라인 재정렬한
결과는 다음과 같다.

```text
score_alpha = proxy^alpha × path_confidence
```

| alpha | 다음 forward에 하나 이상 들어간 root 수 | 현재 선택과 같은 자리 |
|---:|---:|---:|
| 1.00 (현재) | 8.12 / 10 | 100.0% |
| 0.75 | 8.42 / 10 | 95.5% |
| 0.50 (제곱근) | 8.75 / 10 | 91.3% |
| 0.25 | 9.06 / 10 | 87.3% |
| 0.00 (proxy 무시) | 9.21 / 10 | 83.8% |

`alpha=0.5`는 proxy 정보를 버리지 않으면서 극단적 독점을 크게 줄인다.
하지만 평균 1.25개 root는 여전히 첫 깊이에서 멈춘다. 따라서 제곱근만으로
장기 실험의 P2AL 붕괴가 완전히 해결된다고 단정할 수 없다.

## 7. 다음 설계 판단

### 7.1 가장 작은 변경: proxy 제곱근

첫 후보는 `alpha=0.5`다. 근거는 다음과 같다.

- trace 85의 명백히 나쁜 역전을 바로 교정한다.
- 현재 첫-round 선택의 91.3%는 유지해 정책을 과도하게 바꾸지 않는다.
- 과거의 임의 beta sweep과 달리, 실제 score trace에서 확인한 dominance를
  직접 완화한다.

단, 이것은 아직 오프라인 첫-round 반사실 분석이다. 바뀐 부모가 만든 다음
logits/q는 기존 로그에 없으므로 전체 topology와 AL은 짧은 온라인 A/B로
확인해야 한다.

### 7.2 cache coverage를 절대 떨어뜨리지 않을 정책

`R=W=10`에서 한 root의 둘째 부모를 다음 forward에 넣으면 반드시 다른
root 하나는 빠진다. 따라서 "전역 top-10"과 "모든 10개 root의 깊이 보존"을
동시에 만족할 수 없다. 이는 구현 버그가 아니라 lane 예산 제약이다.

cache hit/P2AL 비회귀가 절대 조건이면 더 안전한 정책은 다음이다.

1. 첫 forward에서 10개 root를 모두 평가한다.
2. 이후에도 root마다 정확히 한 부모를 forward한다.
3. 단, 기존처럼 첫 번째로 추출된 자식을 무조건 잇지 않고, 그 root 안에서
   누적 confidence가 가장 높은 자식을 다음 부모로 고른다.
4. `W>R`인 설정에서만 남는 lane을 `sqrt(proxy)×path_confidence` 전역 점수로
   추가 배정한다.

이 방식은 root coverage와 깊이 4를 보존하면서 topology 내용은 confidence에
따라 동적으로 만든다. 현재 `coverage`의 장점과 `eagle`의 장점을 결합하는
정책이며, `alpha=0.5` 단독보다 DUET의 cache 구조에 더 잘 맞을 가능성이 높다.

## 8. 판정

- **구현 완료 여부:** EAGLE식 동적 전역 선택 실행기는 완료.
- **현재 채택 여부:** 미채택. 장기 P2AL/P2 기여가 chain보다 낮았음.
- **proxy dominance:** 항상은 아니지만 극단적 step에서 실제 문제로 확인.
- **제곱근 완화:** 근거 있는 다음 단일 후보. 아직 AL 효과는 미측정.
- **더 근본적인 문제:** root별 결과가 3/8 노드로 갈리는 거친 전역 배분.
  cache coverage를 보존하려면 root별 최소 한 lane을 보장해야 함.
