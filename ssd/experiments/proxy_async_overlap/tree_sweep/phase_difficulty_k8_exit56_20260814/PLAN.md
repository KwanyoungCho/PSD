# Draft-source/Proxy-source/cache-miss difficulty experiment

최종 목적은 TPS 최적화나 tree 비교가 아니라 동일한 speculation 깊이의 chain에서
Proxy-source hit 위치가 Draft-source hit 위치보다 조건부로 어려운지 확인하는
것이다. 코드 내부에서는 각각 P2/P1으로 표기한다. Tree arm과
후속 parameter 탐색은 2026-08-14에 중단했으며 논문 결과에 사용하지 않는다.

## 고정 조건

- `K1=K2=8`, exit layer 56
- LayerSkip Llama-2-70B / TinyLlama-1.1B, batch 1
- native context 2,048, output cap 512, raw prompt
- 기존 cache 정책 유지: P1 fanout 3(root budget 27), P2 budget 15,
  proxy top-k 28, P1 backbone, P2 thresholds 0.01/0.01

과거 tree 진단과 동일한 질문 집합을 유지하기 위해 최종 분석은
`balanced_120q_tree_safe.jsonl`을 사용한다. Eligibility는 결과와 무관하게
`prompt_tokens + 512 + 325 <= 2048`로 정했다. 최초 표본의 긴
summarization 질문 6개를 같은 영역의 짧은 질문 6개로 교체하여 영역당
20질문(총 120질문/140 turns)을 유지한다.

## 순서

1. 2-question smoke에서 phase-event 합과 출력 길이, valid-k, context 경계를
   검증한다.
2. 논문 조건인 temperature 0.7, seed 42에서 `K1=K2=8` chain을 실행한다.
3. seed 42의 질문-bootstrap CI와 subtask 방향을 먼저 본다. 120질문에서
   방향이 일관되고 CI가 0을 넘지 않으면 요청 범위대로 한 seed로 종료하며,
   모호한 경우에만 seed 1/123을 추가한다.
4. 각 step을 P1/P2/miss로 나눠 correction 포함 AL, AL survival curve,
   question-cluster bootstrap CI를 계산한다.
5. 각 temperature-0.7 출력에 TinyLlama teacher-forced replay를 적용해 각
   step의 실제 prefix에서 fresh greedy agreement length와 첫 token NLL/rank를
   계산한다. Cached P2만 낮은지, P2로 분류된 위치의 fresh draft agreement도
   낮은지를 분리한다. 이 값은 stochastic verifier AL이 아니라 source와
   독립적인 draft-difficulty proxy로 명시한다.

Miss는 cache의 남은 어려운 후보가 아니다. 현재 engine은 miss에서 확정 prefix를
사용해 K2-deep JIT draft를 새로 수행하므로, miss AL이 정상적인 것은 예상되는
대조군 동작이다.
