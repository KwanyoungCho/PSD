# DUET tree target-step latency: final cause analysis

Date: 2026-08-12

## Final conclusion

현재 tree의 target-step 증가는 하나의 버그가 아니라 세 가지 비용의 합이다.
소규모 반복 실험으로 현재 설정의 증가량을 거의 전부 설명했다.

| Phase | Current chain width | Current tree width | Profiled conditional tree-hit 증가 | 원인 분해 |
|---|---:|---:|---:|---|
| P1 | K1=8 (target 9 rows) | N1=14 → M1=12 (target 13 rows) | **+6.488 ± 1.112 ms** | 고정 tree machinery +2.472, node 4개 추가 +2.747, 14→12 rerank +1.225 ms |
| P2 | K2=4 (target 5 rows) | N2=M2=8 (target 9 rows) | **+5.214 ± 0.669 ms** | 고정 tree machinery +0.839, node 4개 추가 +4.374 ms |

P1 분해 합은 6.444 ms로 profiled 조건부 직접 측정 6.488 ms와 0.044 ms 차이이고, P2 분해
합은 5.213 ms로 직접 측정 5.214 ms와 0.001 ms 차이다. 즉 현재 관측된 큰
target-step 증가는 아래 경로들로 사실상 모두 설명된다.

세부 event가 tree에 더 많이 들어가므로 이 6.488/5.214 ms를 profiler-off 논문
latency로 사용하면 안 된다. 이 값들은 원인 분해용 조건부 진단이다. Profile-off
전체 tiny7에서는 full-tree − chain이 +4.035 ms였고, 기존 full Spec-Bench raw에서는
chain 65.028 ms 대비 tree 71.028 ms로 +6.000 ms였다.

## Why the original aggregate looked misleading

`target step`은 순수 target forward 시간이 아니다. Draft request/response 대기,
target-side metadata 준비, target verify, acceptance와 scheduler postprocess를 포함한
전체 speculative decode-step wall time이다.

또한 overall 평균은 P1 hit, P2 hit, miss의 비율과 각 출력의 step 수가 섞인다.
Miss에서는 tree와 chain이 거의 같지만 tree hit에서만 5–6 ms 증가하므로, overall
`target verify` 평균만 보면 실제 tree-hit 비용이 작아 보였다. 원인 판정에는 같은
hit source의 조건부 latency가 필요하다.

## Direct critical-path decomposition

동일 tiny7, seed42, output 256을 chain/P2-tree-only/P1+P2-tree 각각 3회
profiled 반복했다.

| Comparison | Full step | Draft/spec wait | Response→verify | Target verify | Post-verify |
|---|---:|---:|---:|---:|---:|
| P1 full tree − chain | +6.488 ms | +2.263 ms | +2.047 ms | +2.162 ms | +0.016 ms |
| P2 P2-tree-only − chain | +5.214 ms | +0.940 ms | +1.532 ms | +2.672 ms | +0.070 ms |
| Miss full tree − chain | +0.528 ms | +0.318 ms | +0.095 ms | +0.115 ms | -0.001 ms |

Miss는 사실상 동일하므로 일반적인 GPU slowdown이나 scheduler 문제가 아니다.
Tree hit에만 존재하는 response/metadata/wider-verify 경로가 원인이다.

## Cause 1: P1 N1=14 → M1=12 hit-time rerank

현재 P1은 14개 node를 생성한 뒤 cache hit가 발생할 때 closure-valid subtree 12개를
다시 고른다. 이 경로는 다음을 수행한다.

1. 생성 tree GPU pack → CPU copy → parse/validate
2. raw-q CPU read와 subtree selection
3. 선택 node의 GPU compaction
4. served tree 재-pack → CPU copy → parse/validate

`14/12`와 `12/12`를 각 3회 반복해 M1을 고정한 결과다.

| Metric | N1=14/M1=12 | N1=12/M1=12 | Difference |
|---|---:|---:|---:|
| P1 rerank span | 1.290 ± 0.158 ms | 0.376 ± 0.094 ms | **-0.914 ± 0.250 ms** |
| P1 cache response | 2.029 ± 0.259 ms | 1.138 ± 0.294 ms | **-0.891 ± 0.545 ms** |
| P1 draft/spec wait | 4.089 ± 0.345 ms | 3.130 ± 0.431 ms | **-0.959 ± 0.762 ms** |
| P1 full step | 70.585 ± 0.468 ms | 69.360 ± 0.269 ms | **-1.225 ± 0.714 ms** |

따라서 `14→12` rerank는 실제 P1 critical path에 약 1 ms를 추가한다. 오류는
아니지만 latency가 큰 구현/정책 trade-off다. N1=M1로 만들면 제거할 수 있으나,
14개에서 좋은 12개를 고르는 효과가 사라져 AL이 달라질 수 있다.

## Cause 2: fixed tree machinery

Target 입력 행 수를 chain과 정확히 맞춘 control로 tree 자체의 고정비를 측정했다.

- P1: chain K1=8/9 rows vs tree M1=8/9 rows
- P2: chain K2=4/5 rows vs tree M2=4/5 rows

| Equal-row comparison | Full step | Draft wait | Pre-verify | Verify |
|---|---:|---:|---:|---:|
| P1 tree8 − chain8 | **+2.472 ms** | +0.941 ms | +1.655 ms | -0.137 ms |
| P2 tree4 − chain4 | **+0.839 ms** | +0.581 ms | +1.167 ms | -1.002 ms |

동일 target row 수에서는 target verify가 느려지지 않았다. 증가량은 draft response와
target pre-verify에 집중된다. P2의 tree acceptance가 같은 폭의 chain acceptance보다
빨라 fixed cost 일부를 상쇄하므로 P2 net fixed cost가 더 작다.

### Draft-side fixed work

Tree hit에는 chain response에 없는 작업이 있다.

- 직전 tree에서 수락된 경로의 draft KV를 canonical slot으로 복원
- served topology pack/CPU validation
- parent-q row gather
- tree metadata가 포함된 response 작성

대표 span은 P1 tree8에서 KV restore 약 0.49 ms, served-tree pack 약 0.34 ms,
parent-q gather 약 0.10 ms다. P2 tree4는 각각 약 0.27, 0.27, 0.13 ms다.

### Target pre-verify fixed work

현재 metadata 경로는 다음과 같다.

1. Fused GPU response에서 valid/phase scalar를 CPU로 읽는다.
2. Tree wire 전체를 `.tolist()`로 CPU에 읽고 parse/validate한다.
3. Parent/sibling list에서 CPU tensor 5개를 매 hit마다 새로 만든다.
4. `child`, `child_valid`, `parent`, `sibling`, `node_valid`를 5번의 작은
   H2D copy로 persistent proxy-graph buffer에 넣는다.
5. Parent-q reference를 다시 GPU에서 select한다.

| Span | P1 M1=12 | P2 M2=8 |
|---|---:|---:|
| Wire list/parse/validate | 0.678 ms | 0.494 ms |
| Topology CPU pack | 0.752 ms | 0.504 ms |
| Five topology H2D copies | 0.241 ms | 0.230 ms |
| Parent-q select | 0.141 ms | 0.137 ms |
| Measured subtotal | **1.812 ms** | **1.365 ms** |

이 subtotal은 직접 측정한 pre-verify 증가 P1 2.047 ms, P2 1.532 ms의 대부분을
설명한다. 데이터는 수십 개 정수에 불과하므로 전송량이 아니라 Python allocation,
GPU→CPU synchronization과 여러 작은 dispatch가 문제다.

## Cause 3: four additional target verification nodes

Current tree는 chain과 같은 proposal 수를 tree 형태로 바꾼 것이 아니라, P1/P2 모두
target에 4개 node를 더 보낸다. 같은 tree implementation에서 node 수만 4개 늘린
control의 결과다.

| Width increase | Full step | Draft wait | Pre-verify | Verify |
|---|---:|---:|---:|---:|
| P1 tree M1 8→12 | **+2.747 ms** | -0.424 ms | +0.408 ms | **+2.740 ms** |
| P2 tree M2 4→8 | **+4.374 ms** | +0.359 ms | +0.365 ms | **+3.673 ms** |

이 비용은 대부분 target verify이며 현재 큰 증가의 가장 중요한 구조적 원인이다.
Tree verify setup과 model path에서 다음 작업이 node 수에 따라 증가한다.

- depth/ancestor와 packed custom attention mask 생성
- input/rope/slot persistent buffer copy
- FlashInfer tree-attention buffer update
- graph pre/post의 추가 token rows
- exit proxy와 final logits의 추가 rows
- tree acceptance walk 및 필요한 경우 scratch→canonical KV commit

세부 CUDA span은 side stream과 nested span이 있어 단순 합산해서는 안 되지만,
high-level `target verify` 경계는 비중첩이므로 위 2.74/3.67 ms가 실제 critical-path
증가다.

## What is not the cause

### Parent-q payload transfer

Target의 Q receive는 chain 약 0.148 ms, tree 약 0.125–0.129 ms였다. Tree parent-q는
유효 row만 보내며 chain K-wide q보다 transfer span이 크지 않다. Target에서 길게
보였던 `recv fused`는 target이 먼저 blocking receive를 게시한 뒤 draft response
준비를 기다린 시간이지 metadata 전송 시간 자체가 아니다.

### Scheduler/postprocess

Post-verify 차이는 약 0–0.07 ms다. Scheduler의 빈 sequence나 일반 postprocess는
현재 tree latency 증가의 원인이 아니다.

### Cache hit rate or AL

Miss 경로는 동일하고 hit에서만 증가한다. Tree는 첫 root 선택/hit 여부를 바꾸는
정책이 아니라 hit 이후 proposal topology와 verify 폭을 바꾼다. 이번 분석은 같은
hit source의 조건부 latency를 사용했으며 hit/AL 편차를 원인으로 보지 않았다.

### General GPU slowdown

Chain/P2-tree/full-tree를 같은 GPU와 입력에서 교차 반복했고 miss는 동일했다. 따라서
tree arm 전체에 걸친 일반적인 GPU slowdown으로 설명되지 않는다.

## Is it a correctness bug?

현재까지 잘못된 token/tree topology, target에 잘못된 row 전달, acceptance 오류의
증거는 없다. 관련 tree contract 테스트 65개도 통과했다. 따라서 주된 문제는
correctness bug가 아니라 다음 두 종류의 성능 문제다.

- 제거 가능한 구현 비효율: 반복 CPU parse/validation, tensor allocation, 작은 H2D,
  draft pack/KV restoration, P1 rerank
- 의도된 구조적 trade-off: AL을 높이기 위해 chain보다 4개 많은 node를 target에서
  검증하는 비용

## Experimental interpretation issue

현재 chain과 tree 설정은 target verify 입력 수가 같지 않다.

- P1: chain 9 rows, current tree 13 rows
- P2: chain 5 rows, current tree 9 rows

따라서 현재 설정으로 얻은 target verify latency를 “같은 양을 검증했는데 tree가 더
느리다” 또는 “같은 verify latency에서 AL만 증가했다”는 근거로 쓰면 안 된다.
동일 입력 수 비교는 P1 chain8/tree8, P2 chain4/tree4 control을 사용해야 한다.
논문 timeline의 (a)/(b)가 같은 target verify 입력 수를 전제로 한다면 이 matched-width
control이 맞는 근거이고, current M1=12/M2=8 실험값은 별도의 AL–latency trade-off
점으로 표기해야 한다.

## Optimization potential

### Priority 1 implementation result (2026-08-12)

Target proxy topology와 verify input/RoPE/mask 준비를 persistent GPU buffer로
옮겼다. CPU fallback은 `SSD_TREE_TOPOLOGY_GPU=0`으로 유지한다. CPU/GPU 출력
hash는 P1/P2 각각 7/7 일치했고, 직접 준비 구간은 P1 tree hit 약 0.56 ms,
P2 tree hit 약 0.33 ms 감소했다. 다만 최종 profiler-off pair는 P1 target
step -0.953 ms, P2 +0.543 ms로 방향이 달랐으므로 end-to-end 차이는 run 편차보다
작고 안정적인 TPS 향상으로 주장할 수 없다.

자세한 구현, correctness, A/B 결과는
[`gpu_topology_ab/GPU_TOPOLOGY_OPTIMIZATION.md`](gpu_topology_ab/GPU_TOPOLOGY_OPTIMIZATION.md)에
정리했다. 따라서 아래 1--4번 중 target 측 GPU 준비는 완료됐지만, 현재 dominant
cost인 추가 verify node 4개는 그대로 남아 있다.

### P1 rerank scheduling result (2026-08-12)

트리 폭/선택 규칙을 유지한 채 `N1=14 → M1=12` rerank만 최적화했다. 모든 P1
root의 결과를 P1 생성 직후 단일 fused CUDA kernel로 미리 만들고, 실제 hit에서는
선택된 wire row의 readback/validation만 수행한다. Legacy fallback은
`SSD_P1_RERANK_PRECOMPUTE=0`으로 유지한다.

- 동일 seed/config의 paired 5회에서 prompt output hash **35/35** 일치
- fused precompute: **0.046 ± 0.000 ms/step**
- P1 hit rerank: **1.781 → 0.268 ms** (`−1.513 ± 0.564 ms`)
- P1 cache response: **2.730 → 0.979 ms** (`−1.751 ± 0.992 ms`)
- profiler-on P1 조건부 wait: **5.115 → 3.177 ms**
  (`−1.938 ± 1.383 ms`)
- profiler-off 전체 target step: `−0.266 ± 1.202 ms`로 run noise보다 작음

즉 P1 hit critical path의 직접 비용은 의미 있게 줄었지만, overall TPS/step 개선은
아직 주장할 수 없다. 추가 verify node와 나머지 tree 공통 경로가 지배적이다.
자세한 결과는
[`rerank_precompute_fused_v2_ab/RESULTS.md`](rerank_precompute_fused_v2_ab/RESULTS.md)에
정리했다.

### Without changing tree width or AL policy

다음은 node 수와 선택 정책을 유지할 수 있는 순수 실행경로 최적화다.

1. Rank0에서 tree wire를 CPU list로 변환하고 다시 tensor로 만들지 않는다.
2. Topology를 하나의 persistent packed buffer로 만들어 5개 H2D를 한 번으로 합치거나
   GPU에서 직접 pack한다.
3. Validated immutable topology 결과를 verifier/proxy/target verify/acceptance에서
   재사용한다. 단, 모든 TP rank가 같은 metadata로 함께 실패한다는 deadlock-safety
   계약은 보존해야 한다.
4. Depth/ancestor/packed mask도 topology와 함께 한 번 계산해 재사용한다.
5. Draft의 served-tree pack/validation과 KV restore에 persistent tensor/buffer를
   사용한다.

현재 측정상 target pre-verify에서만 이론상 P1 약 1.8 ms, P2 약 1.4 ms가 노출돼
있다. 안전 검사와 필수 copy를 완전히 0으로 만들 수는 없으므로 현실적인 회수량은
이보다 작다. Draft fixed work까지 합치면 순수 구현 최적화의 실용적 목표는 tree
hit당 약 1–2 ms다.

### Changing P1 generation/verification policy

`N1=M1=12`는 P1 hit당 약 1.2 ms를 줄일 수 있다. 다만 AL 영향은 full-data에서
검증해야 한다. Rerank를 유지하려면 GPU-native topological selection/compaction으로
CPU round trip을 제거해야 한다.

### Changing verification width

현재 가장 큰 비용은 추가 node 4개다. 이를 줄이는 방법은 다음과 같다.

- M1/M2를 줄인다.
- Confidence에 따라 실제 verify node 수를 step별로 줄이는 adaptive width를 사용한다.
- 추가 node가 기대 acceptance gain보다 verification cost가 큰 경우 확장을 멈춘다.

따라서 threshold/adaptive-node 개념은 latency 관점에서 필요하다. 다만 threshold가
AL을 지나치게 낮추지 않도록 전체 Spec-Bench에서 TPS/AL frontier를 찾아야 한다.

## Profiler perturbation check

세부 profile은 tree에서 더 많은 CUDA event를 기록하므로 절대 latency를 바꿀 수
있다. 동일 triad를 `SSD_PROFILE_DUET=0`으로 2회씩 별도 반복했다.

| Profiler off | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Target step | 67.281 ± 0.762 ms | 68.298 ± 0.070 ms | 71.315 ± 0.110 ms |
| Target verify | 61.189 ± 0.352 ms | 61.439 ± 0.033 ms | 62.342 ± 0.090 ms |
| Outside verify | 6.092 ± 0.409 ms | 6.859 ± 0.037 ms | 8.974 ± 0.199 ms |

Profile-off full-tree − chain target step은 **+4.035 ms**, P2-tree-only − chain은
**+1.018 ms**로 증가가 그대로 남았다. Profiled overall 증가는 각각 +4.604,
+1.430 ms였으므로 상세 event가 absolute gap을 약 0.4–0.6 ms 키웠지만 현상을
만든 것은 아니다. 세부 원인 판정은 profiled 조건부 span, 최종 논문 TPS/latency는
profiler-off full-data 실험만 사용해야 한다.

## Recommended next order

1. Metadata/topology persistent packed-buffer 최적화를 구현한다.
2. 동일 tiny7 matched-width control로 output hash와 latency를 A/B한다.
3. `N1=M1` 후보와 M1/M2 주변 폭을 profile-off smoke에서 비교한다.
4. 그 뒤 full Spec-Bench에서 decode-only TPS, P1/P2 AL, target-step을 측정한다.
5. 최종적으로 C/threshold sweep을 재개한다.

## Artifacts

- Direct triad report: `full_gap_triad/FULL_GAP_ANALYSIS.md`
- Matched-width report: `matched_width_controls/MATCHED_WIDTH_ANALYSIS.md`
- Rerank report: `rerank_ab/LATENCY_AB.md`
- Profiler-off validation: `profile_off_triad/PROFILE_OFF_VALIDATION.md`
- Earlier boundary analysis: `TARGET_STEP_ANALYSIS.md`
- Runners: `run_full_gap_triad.sh`, `run_matched_width_controls.sh`,
  `run_rerank_latency_ab.sh`
- Analyzers: `analyze_full_gap_triad.py`,
  `analyze_matched_width_controls.py`, `analyze_rerank_latency_ab.py`,
  `analyze_profile_off_triad.py`
