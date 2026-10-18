# 점수순 subtree 선택과 target 분포 보존

## 선택할 점수

한 root 후보 r의 후속 tree 노드 v에 대해 `S_r(v)=∏ q(x_u | prefix_u)`를 쓴다. 곱은 r 다음 토큰부터 v까지의 원래 draft 확률이며, 형제 제거 후의 재정규화 확률이 아니다. 구현은 합산 log 확률로 비교한다.

- 확장 단계의 여러 root 간 경쟁은 기존 `root 후보 점수 × q-path` 그대로다.
- 최종 선택은 root별 M개다. 같은 root 안에서는 root 점수가 상수이므로 위 S만 비교해도 순위가 같다.
- 각 변의 q는 0~1이므로 자식 점수가 부모 점수보다 크지 않다. 동점은 생성 순서로 정하면 전역 top-M은 부모를 보존한다. q=0인 경로도 생성 순서로 동점을 처리한다.
- 선택된 노드를 원래 위상 순서로 다시 정렬하고 부모·형제 index를 재작성한다. 앞선 형제를 반드시 보존할 필요는 없다. 이것은 아래 target-sampling 검증에 한정한 설명이다.
- 별도 root별 생성 G를 M에 묶지 않는다. K회 forward와 실제 입력 폭으로 자연스러운 전체 생성 상한 `C×Σ_f W_f`가 정해진다. 검증으로 전송하는 노드에는 최종 M을 적용한다.

## 기존 q를 그대로 p/q에 넣으면 안 되는 이유

후보가 원래 q에서 뽑혔더라도, 그 토큰의 값·후속 분포를 보고 사후 선택하면 선택된 후보의 확률 법칙은 일반적으로 q와 다르다. 부모와 앞선 형제를 보존하는 것만으로 해결되지 않는다.

`test_score_tree_v15.TargetLaw.test_q_equal_p_does_not_save_score_pruned_original_q_verifier`의 유한 열거 반례:

1. p=q=(0.6, 0.3, 0.1).
2. root 형제 2개를 ordered WOR로 뽑고, 첫 형제 아래 자식 2개를 ordered WOR로 뽑는다.
3. 생성 4노드에서 부모·앞선 형제 보존 rerank로 M=2를 고른다.
4. 선택된 토큰에 기존 q의 p/q 검증을 사용한다.
5. 둘째 출력 분포는 `(0.7272857143, 0.2157857143, 0.0569285714)`가 된다. target과 TV=0.1272857143이다.

따라서 기존 검증을 둔 채 점수순으로만 바꾸는 구현은 채택하지 않는다.

## 채택한 검증

target이 tree의 모든 유효 문맥에서 logits를 계산한 뒤, 문맥 c마다 `Y_c ~ p_T(. | prefix+c)`를 독립적으로 뽑는다.

1. root 문맥에서 Y를 확인한다.
2. Y와 같은 토큰의 자식이 있으면 그 노드를 출력하고 해당 문맥으로 이동한다.
3. 자식이 없으면 Y를 recovery로 반환하고 현재 round를 끝낸다.
4. 도달하지 않은 문맥에서 미리 뽑은 값은 버린다.

어떤 draft tree τ가 선택되었든, 현재 확정 prefix와 τ를 조건으로 다음 출력은 정확히 `p_T`에서 뽑은 Y다. 자식의 존재는 이미 뽑은 토큰을 바꾸지 않고, 다음 target forward가 필요한지만 바꾼다. 도달한 자식 문맥에서도 같은 논리를 반복하면 전체 출력의 결합분포가 target의 자기회귀 분포와 같다. T=0에서는 각 문맥의 target argmax를 사용한다.

전제는 정확한 prefix/tree attention, 올바른 KV commit과 독립 target 난수다. BF16 행렬 연산 shape에 따른 수치 차이까지 제거하거나, 같은 seed의 AR과 bitwise 같은 출력을 보장한다는 뜻은 아니다.

## 새 검증에서 경로 점수의 목적

root hit를 조건으로 고정된 tree τ를 생각하고, EOS·출력 상한으로 잘리지 않는 round를 가정한다. 각 부모의 자식 토큰이 서로 다르면 노드 v까지의 경로가 재사용될 확률은 정확히 `P_T(v)=∏ p_T(x_u | prefix_u)`다. 따라서

`E[재사용된 후속 노드 수 | τ, root hit] = Σ_(v∈τ) P_T(v)`.

이는 각 노드를 방문했는지 나타내는 지시변수의 합에 기대값의 선형성을 적용한 결과다. Engine의 root 포함 AL에 대응시키면 1을 더한다. 따라서 **고정된 생성 forest에서 같은 M개를 고르는 문제**는 이 합을 크게 만드는 문제다. p를 미리 알 수 없으므로 q-path로 대리하면 높은 경로 점수의 노드를 고르는 근거가 된다. 경로 점수의 단조성 덕분에 상위 M개가 부모도 보존한다.

단, q≈p는 가정/근사이며 일반적인 정확한 수락확률은 아니다. 두 phase의 시간 예산·제한된 forward 탐색·root hit 확률까지 포함한 전역 최적성을 뜻하지도 않는다. 실제 AL은 반드시 새 실험으로 확인한다.

## 기존 residual 해석에 미치는 영향

새 검증은 rejection residual을 뽑지 않는다. 따라서 기존 `e(1-q)`와 `0.75 h_hat + 0.25 h_bar`는 cache 후보의 우선순위를 위한 기존 휴리스틱으로만 유지한다. 새 verifier의 실제 correction/종료 분포를 정확하게 근사한다고 주장할 수 없다. 후보가 실제로 맞는지는 여전히 `(요청, 문맥, recovery 토큰)` key로 확인한다. 틀리면 miss로 처리한다.

Tree-enabled 실행에서는 B=1, B>1, hit tree와 miss chain 모두 같은 target-sampling 검증을 쓴다. 두 phase를 모두 끈 pure-chain 실행은 기존 p/q 경로를 유지한다. v1.4 대비 AL/hit/TPS 변화에는 선택법과 verifier 변경이 함께 들어가므로, 점수 선택만의 인과 효과로 보고하지 않는다.

## 선행연구와 선택 이유

- [SpecExec, NeurIPS 2024 §4.1](https://papers.neurips.cc/paper_files/paper/2024/file/1d91d5689e251d27993a3c2182dddcf7-Paper-Conference.pdf): 임의 tree에 대해 target에서 직접 sampling하는 캐시 방식이 가능하며, tree의 proposal 법칙이 알려져 있을 필요가 없다. 이번 검증은 이 원리를 따른다. 논문의 동일 seed 출력 동등성까지 이번 병렬 RNG 구현에 가져오지는 않는다.
- [RheoSampling §3.1–3.3, 2026 preprint](https://arxiv.org/html/2609.21827v1): sampled-token의 값에 의존한 pruning이 losslessness를 깨는 문제를 명시한다. 특수하게 설계한 mixed candidate pool, 별도 proxy 순위와 실제 sampling 법칙, OT 검증을 함께 제안한다. 현재의 기존 ordered-WOR tree에 원래 q만 끼워 넣는 것과 다르다.
- [Tsubame §2–3, 2026 preprint](https://arxiv.org/html/2609.33652v1): tree 모양 결정 후 토큰을 다시 sampling한다. 자기회귀 draft에서는 토큰이 바뀌면 후속 forward가 다시 필요하므로 이번 저비용 변경으로 그대로 도입하지 않는다.

이 검증 방식 자체를 DUET의 새로운 발명이라고 주장하지 않는다. 선택의 자유와 단순한 정확성 근거를 얻는 대신, 기존 p/q coupling의 높은 수락률을 잃을 수 있다. 실제 AL/TPS의 개선 여부는 별도 측정 대상이다.
