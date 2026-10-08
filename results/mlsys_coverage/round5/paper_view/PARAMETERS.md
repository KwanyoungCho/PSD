# 사용 파라미터

최종 처리량 우선점과 AL 우선점을 분리한다. Exit는 **0-based**다. N은 root마다 보존하는 continuation 노드 상한, M은 검증 노드 상한, C는 sibling capacity, U는 P1 위치당 root 수, W는 P2 총 root/forward budget이다. 실제 매 step의 유효 노드 수는 이 상한보다 작을 수 있다. SSD K/F는 draft 길이/fanout이다.

공통 평가: full32-layer target, target1GPU+draft1GPU, T=0.7, 입력/출력 cap512/128, 자연 EOS. AL 우선점은 최초 동결 설정을 유지했으므로 최종 처리량 우선점과 trim/stream 옵션도 다를 수 있다.

## Final TPS-priority

| 모델 | B | 방법 | K1/K2 또는 SSD K/F | Exit | C | N1/N2 | M1/M2 | U | DFO/PFO | W | trim/stream |
|---|---:|---|---|---:|---:|---|---|---:|---|---:|---|
| llama2 | 1 | SSD | 4/7 | — | — | — | — | — | — | — | — |
| llama2 | 1 | DUET | 11/2 | 21 | 3 | 16/4 | 16/4 | 2 | 2/1 | 12 | 1/1 |
| llama2 | 8 | SSD | 4/7 | — | — | — | — | — | — | — | — |
| llama2 | 8 | DUET | 3/2 | 21 | 3 | 6/4 | 6/4 | 1 | 2/1 | 4 | 1/0 |
| llama3 | 1 | SSD | 4/7 | — | — | — | — | — | — | — | — |
| llama3 | 1 | DUET | 4/2 | 16 | 3 | 8/4 | 8/4 | 2 | 2/1 | 5 | 1/1 |
| llama3 | 8 | SSD | 4/1 | — | — | — | — | — | — | — | — |
| llama3 | 8 | DUET | 3/1 | 26 | 1 | 3/1 | 3/1 | 2 | 2/1 | 4 | 1/0 |

## AL-priority (original frozen)

| 모델 | B | 방법 | K1/K2 또는 SSD K/F | Exit | C | N1/N2 | M1/M2 | U | DFO/PFO | W | trim/stream |
|---|---:|---|---|---:|---:|---|---|---:|---|---:|---|
| llama2 | 1 | SSD | 6/5 | — | — | — | — | — | — | — | — |
| llama2 | 1 | DUET | 12/2 | 21 | 3 | 16/4 | 16/4 | 2 | 2/1 | 13 | 0/0 |
| llama2 | 8 | SSD | 8/5 | — | — | — | — | — | — | — | — |
| llama2 | 8 | DUET | 6/2 | 21 | 3 | 12/4 | 12/4 | 2 | 3/1 | 7 | 0/0 |
| llama3 | 1 | SSD | 8/1 | — | — | — | — | — | — | — | — |
| llama3 | 1 | DUET | 6/4 | 16 | 3 | 12/8 | 12/8 | 2 | 2/1 | 7 | 0/0 |
| llama3 | 8 | SSD | 8/3 | — | — | — | — | — | — | — | — |
| llama3 | 8 | DUET | 6/3 | 30 | 3 | 12/4 | 12/4 | 2 | 3/1 | 7 | 0/0 |

## 공통 DUET 수식·tree·구현

- 내부 context root: 실제 온도의 `e(1-q)`, leaf는 `e`. P1은 draft 기반 점수.
- 위치: token-conditioned terminal mass 0.75 + overlap terminal mass 0.25. 두 ladder를 각각 계산한 후 혼합.
- 전체 허용 vocabulary 정규화. Verification은 실제 proposal q를 유지.
- `reach_gain_frontier`, 기존 Round4의 reach/gain calibration 고정. 이번 데이터로 재학습하지 않음.
- Miss chain4, P2 proxy/confidence floor=0.01/0.03, P1 start/confidence floor=0/0.
- Fused math=ON, bulk export=ON, parallel insert=OFF. C=1은 root별 continuation이 chain인 경우.
- beta=0.5는 기록되어 있지만 선택한 정책에서는 비활성. 새로운 최적화 결과로 해석하지 않음.
- SSD에도 공통 correctness/fast-verifier/CUDA-graph 경로 적용. SSD 길이/fanout은 독립 튜닝.

## 전체 실험 설정

[ALL_PARAMETERS.csv](ALL_PARAMETERS.csv)에 487개 실험의 설정을 보존했다. 각 실험 HTML 하단에는 원본 parsed args, engine kwargs, env와 effective flags, source hash를 함께 넣었다. 반복별 seed/B/T는 [ALL_PASSES.csv](ALL_PASSES.csv)에서 확인한다. B2/B4·장문 실험은 최초 B8 설정 이전이며 개별 최적점이 아니다.
