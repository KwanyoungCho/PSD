**검증 완료 범위:** full dense 두 모델 pair에서 GPU job94개를 모두 완료했다. 그중62개 job은 전체480질문을 사용했고, target-seed pass는132회다. 전체178개 cell의 실제 반환3,614,764token을 event/step/counter와 대조한1,958개 집계 검사에서 불일치가 없었다. 할당 GPU에 외부 process가 겹친 기록도 없었다. 기본 경로와 fused+bulk+parallel 경로 각각288/288 회귀 검사가 통과했다.

**AL을 우선한 최종 검증 조합:** K1/K2=4/2, 기존 reach+gain+frontier table, miss chain4, fused mask/fanout + bulk export. Root 후보 수식은 그대로다. 아래 비교는 두 arm 모두 같은480질문/target seed2026·2027/draft startup seed0다. AL*는 cap/clip terminal event 제외, TPS*는 경계 batch-step의 token과 시간을 함께 제외했다. TPS는 후속 pass 값을 사용하며 괄호에는 제외 전 실제 반환 TPS를 함께 썼다.

| 모델/B8 | 기존 AL* → 조합 AL* | AL* 개선 | ΔAL* 95% CI | 기존 TPS* → 조합 TPS* (전체 TPS) |
|---|---:|---:|---|---|
| llama2 | 1.9808 → 2.0476 | +3.37% | [+0.0367, +0.0989] | 480.1 → 498.4 (484.5 → 513.3) |
| llama3 | 2.3736 → 2.5376 | +6.91% | [+0.1233, +0.2031] | 381.8 → 422.0 (382.2 → 411.3) |

이 조합의 AL 개선에는 **miss node2→4 확대 효과가 포함**된다. 이를 모두 tree 점수 개선으로 주장하지 않는다. 같은 corpus에서 구성 요소를 본 뒤 확인한 조합이며, 새로운 미관측 test set의 확증도 아니다. 검증한 후보 중 AL 우선 선택지이지 모든 parameter 조합의 최적성 증명은 아니다.

**구성 요소별 결론:**

1. **기존 tree 점수 개선은 Llama3에서 유효했다.** Miss2/G=M8·4를 고정한3pass 비교에서 historical reach+gain+frontier의 AL*는 Llama3 +1.91%(ΔCI [+.0172,+.0746]), Llama2 +0.63%(ΔCI [-.0096,+.0350])다. Llama2의 우위는 확정하지 않는다. Legacy B1에서도 Llama3 +2.72%, Llama2 +1.49%이며 후자는 CI가0을 포함한다.
2. **8질문 dense calibration은 필수가 아니었다.** Dense reach+gain+frontier는 q-path 대비 Llama3 +2.10%, held-out472에서 +2.12%였으나, historical table 자체와 비교한 추가 ΔAL* CI는 Llama3 [-.0277,+.0335], Llama2 [-.0128,+.0305]다. 보정 표본이 작고 수집B1/평가B8의 context 이동도 있어 새 보정이 더 좋다고 확정하지 않는다.
3. **Miss를 무조건 짧게 하거나 넓히는 것은 AL 목표에 맞지 않았다.** 기본 miss는 이미 chain2다. B8에서 chain4의 AL*는 Llama2 1.991→2.030, Llama3 2.382→2.469로 증가했다. 같은4-node의 tree2x2는 각각2.010/2.409로 chain4보다 낮았다. 이는 첫 sibling만 다음 깊이로 확장하는 이번 얕은 설계의 결과이며, 모든 token tree/SpecInfer의 열등성을 뜻하지 않는다. Star는 구조상 AL<=2라 Llama3 chain4 miss 조건부 AL≈2.31을 따라갈 수 없다.
4. **Llama3에서는 miss4에서도 tree 점수의 추가 AL 이득이 남았다.** Target seeds2026·2027로 맞춘 AL-only 분석에서 chain4 조건의 score 개선 ΔAL*는 Llama3 +.0888(CI [+.0499,+.1272]), Llama2 +.0120(CI [-.0182,+.0430])다. 조합 arm의 kernel 구현도 달라 TPS의 완전 factorial 비교는 아니며, 별도로 확인한 구현 동등성 범위에서 AL을 해석한다. 두 요소의 interaction CI는 양 모델 모두0을 포함하므로 양의 synergy를 확정하지 않는다. Llama2의2seed 결과만 골라3seed 주 분석보다 강한 결론을 내리지 않는다.

**알고리즘을 바꾸지 않은 실행 최적화:** K4/2·q-path·miss2의 전체480개 출력이 매 pass 모두 동일했다. Fused/bulk/parallel의 main 비교3개×2모델×2pass, 그리고 독립 draft seed1의 보조 pair 비교에서도 출력 동등성을 확인했다. B1 두 모델의 fused+bulk 비교도 전체480출력과 AL이 동일하다.

| 모델/B8 | 같은 AL을 유지한 관측 최고 설정 | 후속 pass 전체 TPS | 경계 제외 TPS* |
|---|---|---:|---:|
| llama2 | fused | 484.5 → 556.2 (+14.8%) | 480.1 → 551.2 |
| llama3 | fused_bulk_parallel | 382.2 → 456.0 (+19.3%) | 381.8 → 455.0 |

별도 GPU pair·draft seed1/target3030·3031의 fresh-engine 확인에서 fused+bulk 후속 pass TPS는 Llama2 470.0→532.3(+13.3%), Llama3 344.9→398.5(+15.6%)였다. Pair별 절대 TPS는 합치지 않는다. Llama2에서는 bulk/parallel을 더 켜도 fusion 단독보다 빠르지 않았고, Llama3의 parallel 추가 약2% 이득은 별도 process 반복으로 더 확인할 여지가 있다. B1 Llama2에서 같은 최적화의 TPS 이득이 거의 없었던 것은 이미 draft가 숨는 조건과 일관된다. Tree CUDA graph를 매번 다시 만들던 문제를 고친 것이 아니라 graph 안의 작은 kernel/metadata 비용을 줄였다.

**Phase 예산 축소의 원인 분리:** K2/1을 그대로 쓰면 default miss도1로 줄어든다. 따라서 miss2를 고정한 추가 full480 대조를 완료했다. 아래는 같은 fused+bulk, 같은 miss2이며 phase/node 예산만 K4/2·G8/4에서 K2/1·G4/2로 바뀐다.

| 모델/B8 | K4/2 AL* → K2/1 AL* | AL* 변화 | 후속 pass 전체 TPS |
|---|---:|---:|---:|
| llama2 | 1.9808 → 1.7501 | -11.64% | 549.9 → 545.3 |
| llama3 | 2.3736 → 2.0497 | -13.65% | 446.0 → 508.0 |

Llama3에서는 짧은 phase가 TPS를 더 높일 수 있지만 AL을 희생한다. **사용자가 정한 AL 우선 목적에는 K4/2 조합을 유지하는 쪽이 맞다.** Profile에서 두 phase가 시간 안에 들어온다는 조건만으로 최적 parameter가 정해지지 않는다. K를 바꾸면 depth/node/query shape와 target latency도 바뀌므로 이들을 포함한 AL/시간 비교가 필요하다. 여기서는 node budget도 함께 바뀌는 실제 설정을 비교했으며, 모든 K/N 조합을 sweep한 것은 아니다.

**적용 지침:** G>M proposal-law 수정과 실제 반환/경계 집계는 공통으로 채택한다. AL 우선 실행은 `*_budget_plan.json`의 `*_combined_chain4` job을 사용한다. 동일 AL에서 TPS를 우선하는 K4/2 실행은 Llama2의 `*_fused`, Llama3의 `*_fused_bulk_parallel` job이 이번 관측 최고다. 새 성능 옵션의 기본값은0으로 남겨 원 설정 재현/새 하드웨어 대조가 가능하게 했고, 검증한 plan은 필요한 옵션을 명시한다. 이 옵션들은 새 확률 threshold가 아니라 구현 ablation용이다. 다른 서버에서는 위 preset으로 시작하고 backend/hardware 변경 후 parity와 performance를 재확인한다.

**남은 범위:** 이 서버의 위94개 실행과 correctness/집계 검증은 완료했다. Dense70B·Blackwell targetTP2, 긴 출력1024/장문 전체 입력, 새 root 수식과의 결합, 다른 workload·독립 seed에서 작은 AL 이득의 재현은 후속이다. 이번 결과는 신규 SSD/Mirror-SD baseline 비교가 아니므로 그 대비 우위를 이 수치만으로 주장하지 않는다.
