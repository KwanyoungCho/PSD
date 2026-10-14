# DUET tree 확장 구현 결정

`MLSys-ver0.2@1b93104`를 기준으로 `feat/duet-qpath-expansion`에서 확정하고 `MLSys-ver1.0`에 이식한 구현이다. 일반 실행에서는 calibration 없이 **후보 점수 × draft 경로 확률**로 경쟁한다. 확장 대상만 `last`와 `frontier` 중 선택한다. 기본은 `last`이며, 두 방식의 성능 우열을 확정했다는 의미는 아니다.

## 실험에서 결정할 수 있는 내용

근거는 [동일 생성 예산의 14개 실행](results/mlsys_ver02/equal_budget_20261014/FINAL_REPORT.md)이다. LayerSkip 7B + AMD 135M, FP16, K1/K2=8/8, T=0.7, phase별 root 8개·forward 폭 8개, 출력 1024에서 비교했다. 서로 다른 서버와 seed 두 반복의 문항 평균이다.

| 방식 | 전체 AL | chain 형태 대비 |
|---|---:|---:|
| C1 chain 형태 | 3.437 | 기준 |
| Last C3/M32 | 3.640 | +5.91% |
| Frontier C3/M32 | 3.693 | +7.44% |
| Frontier C8/M32 | 2.925 | −14.90% |
| Frontier C8/M64 | 3.694 | +7.49% |

C3는 좋은 출발점이고, C8/M32에서는 검증 prefix가 깊이를 잘랐다. Frontier−Last의 AL 차이는 +0.0524이지만 95% 구간은 [−0.059, +0.161]이다. **두 확장 방식은 옵션으로 유지하며, 점수식·calibration·rerank를 추가 탐색 축으로 늘리지 않는다.** C3가 모든 모델에서 최적이라는 결론은 아니다.

이 실험은 DS 시작 후보를 전체 위치에서 상위 8개로 선정했다. 이번 native 구현은 기존의 **DS 위치별 균등 선정**을 유지한다. 따라서 위 수치를 새 native 설정이나 70B의 실측 성능으로 인용하지 않는다. 기존 실험은 최종 출력 한도에 의해 잘린 step을 AL 집계에서 제외하며, 일반 논문 성능 집계와도 구분해야 한다.

## 공통 알고리즘

1. DS 또는 PS가 선정한 시작 후보와 점수를 받는다. 두 phase의 후보를 한 번에 경쟁시키지 않는다.
2. 첫 forward에서 모든 유효 시작 후보를 평가한다. 물리 입력 폭은 시작 후보 수 이상이다.
3. 평가한 부모마다 현재 draft 분포에서 중복 없이 순서대로 최대 C개 자식을 샘플링한다. 기본 C=3이다. 실제 샘플링 온도와 원래 draft 확률을 사용하며, 형제 제거 후 재정규화한 확률로 경로 점수를 바꾸지 않는다.
4. 아직 forward하지 않은 노드 중 아래 범위에서 후보 점수 × q-path가 높은 노드에 다음 forward 입력 행을 배정한다. 동점은 생성 순서로 처리한다.
5. phase별 K회 forward 또는 유효 후보 소진까지 진행한다. 실행기는 고정 크기 graph를 사용하므로 후보가 부족하면 남은 행은 padding이다.
6. cache hit된 후보의 tree를 사용한다. 검증 상한 M보다 많으면 **생성 순서의 앞 M개**만 남긴다. 점수에 따른 사후 rerank는 하지 않는다. 부모와 먼저 샘플링된 형제의 순서를 유지하며, verifier의 확률과 보정 절차는 바꾸지 않는다.

수식 렌더링 없이 읽을 수 있는 표기는 다음과 같다. i는 후보 토큰이 들어갈 위치이고, y는 이미 뽑힌 draft 경로다.

```text
DS 후보 점수 C_DS(i,v) = [q_0(y_1) × ... × q_(i-1)(y_i)] × q_i(v)
PS 후보 점수 C_PS(i,v) = 위치 점수 H_i × 토큰 점수 R_i(v)

후속 노드 x_d의 점수 = C_phase(i,v) × q(x_1 | i,v) × ... × q(x_d | 이전 경로)
```

이 값은 target acceptance의 정확한 확률이 아니라 **draft 계산 배분용 점수**다. 실제 검증은 원래의 target/draft 분포를 사용한다. PS의 혼합 위치식, e(1−q)와 DS 중복 제외는 ver0.2에서 유지했다. ver1.0에는 연구 checkout의 top-M 정규화를 기본으로 적용하고, full 정규화는 비교 옵션으로 남겼다. [통합 설정](MLSYS_VER1_INTEGRATION.md)을 참고한다.

| 선택 | forward 번호 f에서 경쟁하는 노드 | 특징 |
|---|---|---|
| `last` 기본 | 깊이 f인 미확장 노드 | 직전 round가 생성한 다음 깊이만 전진 |
| `frontier` | 깊이 f 이하의 모든 미확장 노드 | 이전에 밀린 얕은 노드도 다시 경쟁 |

시작 후보의 깊이는 0, 첫 forward 번호도 0이다. 이미 forward한 노드는 다시 확장하지 않는다. 어느 방식도 특정 후보의 첫째 자식이나 주경로에 forward 행을 강제로 보장하지 않는다. **Last도 후보 간 전역 경쟁**이며, 후보별 독립 beam search가 아니다.

## 생성 상한과 깊이의 관계

생성 상한 G와 검증 상한 M은 다르다. 기존 native 실행기의 후보별 G와 문맥 보호를 유지한다. G가 작은 경우에는 각 root의 남은 공간에서 후속 round를 위한 공간을 남기고, 이번에 선택된 부모들에게 한 자식씩 먼저 배정한 뒤 최대 C개까지 배정한다. 이 공간 제한은 특정 주경로에 forward를 보장하는 것과 다르다.

따라서 **이번 구현에서 C는 최대 자식 수**다. 14개 실험처럼 G가 충분하고 draft 분포의 양수 확률 토큰이 C개 이상이면 부모마다 C개를 생성하지만, 작은 G에서는 적게 생성할 수 있다. 이 제한을 숨기고 “모든 실행이 매 부모당 정확히 C개 생성”이라고 쓰면 안 된다.

기존 실험의 고정 C와 생성 순서 prefix에서는 깊이 d의 노드를 남기기 위한 필요조건이 다음이다.

```text
M >= C × (d − 1) + 1
```

K=8, C=8, M=32라면 검증 가능한 깊이는 최대 4다. C3/M32는 이 조건 때문에 깊이 8이 막히지는 않는다. 이것은 충분조건이 아니며, 자식 수가 G에 의해 줄어든 native 실행에 그대로 적용하는 보편 상한도 아니다.

K1/K2, DS·PS forward 폭, 생성 G, 검증 M, exit layer는 **70B의 지연과 overlap을 확인할 시스템 파라미터**다. 이번 구현에서는 7B의 K=8/M32를 70B 기본값으로 승격하지 않았다. C는 기존 기본값 3을 유지했다. 기존 phase on/off 설정도 유지하므로 DS와 PS 모두 tree로 평가할 때는 둘 다 `on`을 지정한다.

## 사용법

기존 모델·GPU·예산 인자에 다음을 추가한다.

```text
--duet_p1_tree_policy on --duet_p2_tree_policy on
--duet_tree_expansion last
```

전체 말단 경쟁은 `--duet_tree_expansion frontier`로 바꾼다. phase마다 다르게 비교하려면 다음처럼 설정한다.

```text
--duet_tree_expansion last --duet_p2_tree_expansion frontier
```

Python API에도 `duet_tree_expansion`, `duet_p1_tree_expansion`, `duet_p2_tree_expansion`을 동일하게 전달한다. phase override를 생략하면 공통 설정을 따른다. 이전 `duet_tree_policy=frontier`는 별도의 legacy 구현 이름이므로 새 옵션 대신 사용하지 않는다.

새 기본 설정은 root 점수·개별 draft 확률의 확장 threshold를 모두 0으로 둔다. 예전 launcher가 threshold를 명시하면 그 값은 유지된다. 이전 설정과 비교할 때는 threshold까지 동일하게 명시해야 확장 범위만의 비교가 된다.

`SSD_TREE_EXPANSION_POLICY`는 unset 또는 `q_path`로 둔다. 과거 calibrated 모드가 환경에 남아 있으면 오류로 알려준다. 이전 연구 재현만 `SSD_TREE_LEGACY_CALIBRATION=1`로 명시적으로 허용한다. calibration 표와 과거 재현 코드는 삭제하지 않았지만 일반 실행의 옵션은 아니다.

## 구현과 검증 위치

- [설정 해석](ssd/ssd/engine/helpers/tree_expansion_config.py), [점수 정책](ssd/ssd/engine/helpers/tree_expansion_policy.py), [Config](ssd/ssd/config.py), [실행 CLI](ssd/bench/bench.py).
- captured P1/P2 및 batched executor는 같은 설정을 읽는다. CPU reference와 eager arena에도 동일한 확장 범위를 연결했다. graph 안에 새 host 동기화나 calibration 추론을 추가하지 않았다.
- [회귀 검사](ssd/tests/test_tree_expansion_modes.py), [ver1.0 검증 결과](results/ver10_integration_20261015/topm_release_checks/CHECKS.json), [소스 및 실행 환경](results/ver10_integration_20261015/topm_release_checks/CHECKS_PROVENANCE.json).

이식 전 tree 전용 브랜치에서는 14개 suite / 248개 검사를 통과했다. 이 수치는 ver1.0의 새 실행 수치와 다르다. [ver1.0 검증 보고서](results/ver10_integration_20261015/REPORT.md)에 통합본의 검사 범위와 결과를 정리한다.

현재 서버의 70B full inference에서 AL·TPS 개선을 새로 측정한 결과는 이 구현 검사와 별개다. 기존 연구 checkout, 두 review worktree 및 기존 실험 결과는 수정하지 않았다.
