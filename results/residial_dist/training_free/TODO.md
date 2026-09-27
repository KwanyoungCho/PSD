# 설명 재개 TODO — 사용자 이해 확인 전까지 미완료

2026-09-13 사용자 요청: 기존 수식 설명을 아직 이해하지 못했으므로, 나중에 전체를
연결해서 다시 설명할 수 있도록 반드시 남긴다. 문서를 읽었거나 설명을 한 번
제공했다는 이유만으로 완료 처리하지 않는다.

2026-09-21: 다른 작업 이후 전체 설명과 연구를 재개하기 위한
[통합 진행 기록](../../DUET_PROGRESS.md)을 만들었다. 후보 수식·결과와 `e(1-q)`의
논문 근거에 대한 설명은 제공했지만, 전체 이해 확인은 아직 끝나지 않았다.
아래 체크박스는 이번 기록 작업에서도 미완료로 유지한다.

9/21 tree 작업 추가: [Tree 보고서](../../duet_tree_analysis/REPORT.md),
[수식](../../duet_tree_analysis/THEORY.md). 요청에 따라 새 tree 실험은 수행했지만,
아래 설명 이해 확인은 완료 처리하지 않는다.

- [ ] 최신 지시인 root→cache hit, tree→AL 목표를 먼저 설명한다.
      [Full Spec-Bench 평가](../../duet_tree_al_full/REPORT.md)와
      [목적함수 해설](../../duet_tree_al_full/OBJECTIVE.md)을 함께 확인하고,
      같은 예산에서 AL을 판정하며 TPS 증가를 필수 조건으로 두지 않는다.
- [ ] [9/22 tree 사후 분석](../../duet_tree_posthoc/FINDINGS.md)을 함께 설명한다.
      기대 AL=reach 합의 타당성과, frozen alpha0>.5 때문에 같은 부모의 첫 형제를
      항상 우선하는 표현력 한계를 구분한다. 약14.22%의 관측된 oracle 역전 사례를 설명한다.
- [ ] MSE 15.03% 개선, 더 큰 자료로 보정했을 때의 작은 선택 개선, 같은8문항에서
      depth/entropy 추가의 실패를 구분한다. MSE 감소를 실제 tree AL 증가로 치환하지 않는다.
- [ ] 후보 수별 gamma(c), 두 부모에 하나씩 줄지 한 부모에 둘을 줄지의 추가 AL 비교,
      residual/WOR의 비오목 반례를 설명한다. 국소 gain +2.95%와 online AL 미검증을 명확히 구분한다.
- [ ] Coin 변동, 현재 feature의 정보 부족, 배분 알고리즘, 미확장 가지의 관측 부족을
      나누고 oracle gap 전체가 회복 가능한 이득은 아닌 이유를 설명한다.
- [ ] 9/22 완료한 3,360턴의 full 결과: AL +1.70%와 CI에 0 포함, P1 깊이 3/4
      수락 비율 증가, P2 불확실성, 실제 hit rate 변화 및 128-token cap의 범위를
      설명한다. 이전 소규모 +9.82%와 full 결과를 구분한다.
- [ ] Root 후보 점수와 tree continuation 점수를 구분하고, q 경로 곱 대신
      parent 도달·앞선 형제 거절·현재 형제 수락을 곱하는 이유를 설명한다.
- [ ] 기대 AL=모든 node reach 합의 유도, 확장 이득에 continuation gain이
      추가로 필요한 이유, scalar calibration의 정확도 한계를 설명한다.
- [ ] G>M 사후 rerank에서 closure만으로 lossless가 보장되지 않는 완전 열거
      반례와, sampling 이전 예산 결정/G=M의 차이를 설명한다.
- [ ] Tree score의 MSE 19.27% 개선, 계측 실행의 AL 감소, untraced 2-seed의
      AL 증가와 TPS +1.91%(CI에 0 포함)를 구분해서 설명한다. Proxy 후보 개선·
      이전 calibration TPS 결과와 혼합하지 않는다.

- [ ] 전체 연구 목표: residual 후보 선정이 proxy-only보다 유리해지는 조건과,
      현재 실제로 입증한 범위를 먼저 설명한다.
- [ ] p(target), q(draft), e(proxy), 실제 correction R, 후보 점수 s의 차이를 설명한다.
- [ ] 위치 내부의 토큰 순위와 위치 사이의 전체 root 예산 배분을 구분한다.
- [ ] 실제 첫 거절 확률 h와 proxy 추정 h_hat의 누적곱을 작은 예로 설명한다.
- [ ] sum min(e,q)=1-TV(e,q)의 기대값 유도와, 여기서 만든 h_bar가 실제 경로의
      정확한 거절 분포는 아니라는 경계를 설명한다.
- [ ] h_tilde=0.75 h_hat+0.25 h_bar: h를 섞는 이유, alpha 혼합과의 차이,
      혼합 계수가 실험으로 선택된 값이라는 점을 설명한다.
- [ ] top-M 재정규화와 전체 허용 vocabulary 정규화의 차이, 후보 예산 영향과
      정확한 입력에서의 최적성/근사 입력에서의 한계를 설명한다.
- [ ] [e-q]+의 절대 오차, 상대 민감도, clipping, 정규화 오차를 구분한다.
- [ ] 부드러운 할인 e(1-q)^beta와 floor max([e-q]+,rho e)의 설계 근거,
      증명되는 민감도 제한과 증명되지 않은 후보 정확도 우위를 구분한다.
- [ ] e(1-q)=[e-q]+ + min(e,q)-eq의 의미와, 독립 불일치 확률이 실제 SD 거절
      확률은 아니라는 점을 설명한다. e=p에서도 residual로 돌아가지 않는 반례를
      함께 보고, 이 식을 논문의 핵심 방법으로 주장할 근거가 충분한지 논의한다.
- [ ] 기존 proxy / 동일 배분의 개선 proxy / 수정 residual을 같은 조건으로 비교한다.
      초기 +0.125%p source 이득이 실제 전송/동점 검사 후 유지되지 않은 결과도 포함한다.
- [ ] coverage, 실제 cache hit, AL, output TPS, Mirror-SD 전체 시스템 비교의 차이를 설명한다.
- [ ] 후보 점수 변경이 실제 correction sampler와 p/q verification을 바꾸지 않는 이유를 설명한다.
- [ ] 이번 calibration 연구의 exit / P1 / P2 / root / tree / 후보 점수 파라미터를
      위 수식과 연결해 전체 흐름을 다시 설명한다.
- [ ] 후보 coverage 개선과 calibration의 +42.19% TPS를 분리한다. 후자는 기존
      residual 후보를 유지한 K1/K2 조정 결과이며 Mirror-SD 대비 결과가 아니다.
- [ ] 설명을 마친 뒤 새 tree 선정 방식, 후보 추정, calibration, 실제 시스템 비교의
      후속 순서를 사용자와 정한다. 새 실험은 사용자 요청 범위에 따라 진행한다.

관련 근거: [THEORY.md](THEORY.md), [REPORT.md](REPORT.md),
[이전 직접 비교](../direct_comparison/THEORY.md),
[후속 분석](../shared_review/FOLLOWUP.md).

Calibration feasibility 실험은 완료되었으며 상세 결과는
[DUET calibration](../../duet_calibration/README.md)에 있다. 설명 재개와 후속 연구의
현재 상태는 [통합 진행 기록](../../DUET_PROGRESS.md)을 먼저 읽는다.

- [ ] Tree 후속 C=3: phase 공통 gamma(1..3), 정수 partition 배분 최적성, 현재 multi-round reserve의 의미를 함께 설명한다. `../../duet_tree_followup/THEORY.md` 참조.
- [ ] 이전 두-node +2.95%와 C=3 observed-round +1.66%(CI에 0 포함)를 구분하고, online AL 결과가 나온 뒤 전체 정책을 비교한다.
- [ ] Draft-only shadow q 재현 gate 실패와 동일 engine/KV 상태 causal replay의 남은 한계를 설명한다.
