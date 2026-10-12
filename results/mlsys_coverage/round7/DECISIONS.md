이번 검토의 기준은 원 논문 commit `a82f7d2`의 실제 실행이다. Round6의
`graph OFF`는 현재 unified 실행기의 옵션이었으므로, 그 개선율을 원 논문
코드 대비 개선율로 설명한 것은 부정확했다. 이번에는 두 기준을 분리했다.

**선별 이식 원칙**

| 항목 | 판정 | 이유 / 이식 범위 |
|---|---|---|
| 양의 온도 chain fast | 이식 추천, 사용자 채택 검토 대상 | 원본과 같은 출력·AL·hit를 유지하며 전체 verification latency 감소. `01_chain_fast.patch`만 가져올 수 있음 |
| 최소 B1 tree graph 후보 | 이식하지 않는 것을 추천 | 두 full480 대조에서 verification 변화 방향이 뒤집히고 TPS도 개선되지 않음. `02`는 재현용 후보로만 보관 |
| 현재 unified tree 실행기 전체 | 실제 이득 확인, 최소 수정과 분리해 채택 검토 | 원 논문 B1 대비 PIX verification−2.26%/TPS+5.47%, NODE−2.75%/+6.66%. RNG·AL·통신/실행 경로도 달라진 묶음 비교. `02`와 다른 구현 |
| 공통 KV 정확성 수정 | 정확성 수정으로 이식 추천, 속도 향상 기여로 주장하지 않음 | 원본 결함4개 재현→수정 후 통과. 일반 decode latency 차이는 반복마다 방향이 달라 실질적 개선이라고 볼 수 없음. `03_kv_correctness.patch` |
| Packed chain/tree, 새 KV promotion | 이번 최소 이식에서 제외 | 별도 실행 경로/자료구조 변경. Chain fast의 효과와 분리해야 하며, 이번 실험은 이를 필요조건으로 사용하지 않음 |

01은 CPU에 이미 있는 온도 정보로 확률 검증 경로를 선택하고, 필요 없는
greedy 계산·GPU 조건 확인·여러 번의 결과 복사를 줄인다. Target forward나
draft/root/tree 정책을 바꾸지 않는다. SSD와 DUET가 공유하는 코드이므로
향후 baseline에도 동일하게 적용해야 한다. DUET만의 알고리즘 기여로 세면
안 된다. T=0·mixed-temperature·non-JIT 경로는 이 패치에서 원본을 유지한다.

02의 graph는 원본의 방문 순서와 난수 사용을 유지하려고 각 노드의 조건부
residual 후보를 미리 계산한다. 원본은 실제 방문한 경로만 계산한다.
Graph replay 자체가 빨라도 추가 후보 계산, metadata 준비·전송, CPU walk가
남는다. 이 비용 구조가 있어 micro의 일부 이득을 전체 tree verification의
이득으로 바로 옮길 수 없다. 실제 full480에서 얻은 결론은 **이 후보의
전체 latency 개선이 재현되지 않았다**는 것이며, 모든 tree graph가
불가능하다는 뜻은 아니다.

현재 unified 실행기 전체는 원본 대비 이득이 관측됐다. 다만 이것을
`02`의 성능으로 설명하거나 graph 하나의 단독 효과라고 부르면 안 된다.
Full480에서 phase를 나눠 보면 miss chain 검증은 약0.97–1.08ms 감소했고,
P1 hit는 거의 같았으며 P2 hit는 약0.11–0.25ms 감소했다. Target forward
전체를 다시 최적화한 효과라기보다 acceptance·요청 준비 등 실행 경로의
차이가 큰 것으로 해석한다. 정확한 비용 귀속은 상세 profile을 함께 본다.
01만 적용한 tree workload 대조가 어느 이득까지 더 작은 변경으로 얻을
수 있는지 보여준다. B>1 tree 기능이 필요하면 unified 실행기를 검토하되,
기존 B1만의 최소 이식을 원하면01과03부터 개별 검토하는 것이 합리적이다.

**최소 패치의 실제 tree workload 결과도 확인했다.** 01만 물리적으로
적용한 원본 tree 코드는 두 GPU 연결 조건에서 모든 출력·AL·hit를 유지했고,
verification 평균17.210→16.969ms(−1.40%), TPS+1.35%였다. Miss chain
구간에서0.59–0.70ms 줄고, P1/P2 hit 구간 변화는 약±0.05ms 이하였다.
원본 miss 비중이 약36%이므로 `0.36 × 0.65ms ≈ 0.23ms`가 전체 step의
verification 절감량으로 예상되며, 관측된 약0.24ms와 맞는다. 이 때문에
chain만 쓰는 workload의 약3.4%와 tree workload의 약1.4%를 구분한다.
이 두 후보는 원본 대조군을 공유한 후속 확인이므로 새로운 동일-topology
독립 반복2회로 세지 않는다.

03은 prefix KV의 page table과 마지막 query, preemption 출력 경계,
완료 block hash를 바로잡는다. 기존 확정 경로의 tree glue/stage/restore를
그대로 사용한다. Prefix 재사용의 prefill 이득과 tree glue 교체의 이득은
별개다. Paged attention 수정의 직접 대상은 SGL prefill backend이며,
기존 Blackwell FlashInfer의 `run_paged` 분기는 이미 따로 있었다.
이번 결함 재현이 과거 논문 run 전체의 출력 오류를 뜻하지는 않는다.

**원 논문 branch에 자동 반영하지 않았다.** 01/02/03을 별도 review checkout에
물리적으로 적용해 integration 검사를 했으며, 실제 채택은 이 보고서의
결과와 patch diff를 검토한 뒤 결정한다. 이미 승인한 실제 온도·G>M·집계·
Scheduler B·wire 변경은 이번 verification/KV 성능 판정과 분리한다.

**이 서버에서 확정할 수 없는 범위**

- 원 논문70B/Blackwell TP2에서의 개선 폭. 같은 최소 patch와 대조 도구로
  해당 장비에서 확인해야 한다. 다른 모델·통신·backend의 효과 크기를
  RTX4090 dense7B/8B 수치로 대신하지 않는다.
- 원본이 지원하지 않았던 B>1 dynamic tree의 원 논문 대비 latency 감소율.
  Round6의 B>1 graph OFF→ON은 새 기능 안에서의 최적화 검증이다.
- 원본 B1 tree 직접 대조는 Llama2 pair다. Llama3의 큰 vocabulary tree는
  원본 wire의 token 범위 제한 때문에 모델 호환 수정이 먼저 필요하다.
  Llama3 tree의 현재 branch 내 근거는 Round6에 있으며, 이를 이번 원본
  checkout과의 직접 대조로 표현하지 않는다. Llama3 chain은 직접 대조했다.
- 전체 서비스의 continuous batching·긴 출력·자연 prefix hit율에서의 TPS.
  이번에는 기존 요청을 B개씩 묶은 고정 decode 대조이며 저장된480입력,
  입력512/출력96 범위를 모두 처리했다. Cap/clip 또는 partial-B step은
  사용자의 집계 기준에 따라 성능 표에서만 제외했다.
