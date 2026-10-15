# MLSys ver1.1 파라미터 정리와 생성 크기 분석

기준: `MLSys-ver1.0`의 `82d964e`. 작업 브랜치: `MLSys-ver1.1`. DS=P1, PS=P2이다.

구현 커밋: `fe699aa5b88e7e5b8cb11ce04715af84e02f31c6`. 최종 검증은 이 커밋의 수정되지 않은 runtime·bench·test 소스로 수행했다. 문서와 증거는 별도 후속 커밋으로 보관한다.

사용자 결정에 따라 **생성 상한 G와 검증 상한 M을 같은 값으로 사용**한다. DS·PS 입력 폭을 정수로 받고, 위치별 후보 추출 개수와 전송 여유분을 자동 계산한다. 확장 threshold는 0으로 고정한다. 이 변경은 파라미터 계약 정리이며 TPS·AL 개선의 실험 결과로 취급하지 않는다.

## 사용할 파라미터

| 의미 | Python 및 CLI 이름 | 70B 시작 preset |
|---|---|---:|
| Early-exit layer | `duet_exit_layer` | 56 |
| DS forward 횟수 | `duet_phase1_k` | 8 |
| PS forward 횟수 | `duet_phase2_k` | 4 |
| DS 위치당 root 수 | `duet_p1_roots_per_position` | 3 |
| PS 전체 root 수 | `duet_p2_root_count` | 10 |
| DS 반복 확장의 입력 수 상한 | `duet_p1_forward_width` | 39 |
| PS 매 forward 입력 수 | `duet_p2_forward_width` | 15 |
| DS root당 생성·검증 노드 수 | `duet_p1_tree_verify_nodes` | 12 |
| PS root당 생성·검증 노드 수 | `duet_p2_tree_verify_nodes` | 8 |
| 부모당 최대 자식 수 C | `duet_tree_c_tensor` | 3 |
| 확장 범위 | `duet_tree_expansion` | `last` |

이 값은 70B의 시작 설정이며 최적 파라미터라는 뜻이 아니다. 일반 Config 기본값과 preset은 구분한다. 온도·모델·dtype·batch·입출력 길이는 기존 실험 설정이다. Root 수식과 위치 혼합 비율은 기존 ver1.0 설정을 유지한다.

### 새 실행 코드에서 지정하는 방법

70B 실험은 기존 데이터 로더·출력 설정을 유지하고 다음 preset을 가져온다. 아래에는 파라미터 변경 위치만 표시했다. Llama2-7B에 exit56을 그대로 쓰면 안 된다.

```python
from ssd import LLM
from ssd.presets import duet_70b_paper_kwargs

settings = duet_70b_paper_kwargs(draft_path)
settings.update(
    duet_phase1_k=8, duet_phase2_k=4, speculate_k=12,
    duet_p1_roots_per_position=3, duet_p2_root_count=10,
    duet_p1_forward_width=39, duet_p2_forward_width=15,
    duet_p1_tree_verify_nodes=12, duet_p2_tree_verify_nodes=8,
    duet_tree_c_tensor=3,
)
llm = LLM(target_path, **settings)
```

Python과 `ssd/bench/bench.py` CLI는 같은 이름을 사용한다. CLI에서는 K1/K2를 지정하고 `--k`를 생략하면 합을 자동 계산한다. 과거 결과를 만든 `mlsys_equal_budget_adapter.py` 등 연구용 실행기는 ver1.0에서 사용한다. G>M 실험기를 새 Config에 억지로 연결하지 않는다.

### DS 입력 수의 정확한 계약

첫 DS forward는 모든 root를 평가한다. 문맥 수 P, 위치당 root 수 U이면 첫 입력 수는 R=P×U이다. 이후 입력 수는 `min(R, duet_p1_forward_width)`이며, 값을 생략하면 R이다. 예: R=18, 입력 상한12, K1=4이면 **18→12→12→12**이다. 작은 문맥을 최대 폭까지 padding하지 않고 기존 버킷별 실행 방식과 root coverage를 유지한다.

이는 모든 round가 지정한 폭으로 고정된다는 계약이 아니다. PS는 R≤W를 검사하고 매회 W행을 사용한다. Root별 확장 할당과 유효 행 수는 계속 점수·남은 노드 수에 따라 결정된다. `backbone`을 명시할 때 DS 입력 상한이 root 수보다 작으면 오류로 처리한다.

70B preset의 종전 scale1.5는 긴 chain 폭27에서41을 만든 뒤 root 수로 잘랐다. 이번 상한39는 해당 preset의 최대 root 수39이며, 기존 모든 P1 문맥 버킷에서 같은 실제 round 폭을 낸다. 다른 모델의 scale을 이 숫자로 일괄 치환하면 안 된다. 기존 `[tree-policy-resolved].round_widths`를 보고 직접 입력 상한을 정해야 한다. 특히 일반 Config에서 입력 상한을 생략한 동작은 옛 scale1.0과 항상 같지 않다.

## 제거 및 자동화

| 이전 입력 | ver1.1 처리 |
|---|---|
| DS forward scale | 제거. `duet_p1_forward_width`로 직접 지정 |
| PS budget | `duet_p2_forward_width`로 이름 정리 |
| PS 활성 시작 후보 상한 | `duet_p2_root_count`로 이름 정리 |
| 생성 상한 G | 독립 입력 제거. phase별 검증 노드 M에서 자동 결정 |
| 위치별 top-M 입력 | 자동 top-L. L은 전송 후보 수 N과 같음 |
| DS·PS 확장 threshold 4개 | 독립 입력 제거. 0으로 고정 |
| 전송 여유분 | DS 후보 중복의 최대 개수에서 자동 산출 |

최종 PS root 수 R에 DS 중복 여유 D와 기존 고정 여유2를 더해 N개를 전송한다. Target은 위치마다 상위 L=N개를 뽑고 위치·토큰 점수로 전체 상위 N개만 전송한다. Draft는 실제 DS 후보와 중복을 제거하고 R개를 확정한다. **위치 개수×N개 전체를 전송하지 않는다.** 70B preset은 기존과 같이 N=51을 보내며, 위치별 추출 수만28→51로 바뀐다.

**topm 정규화의 분모도 L에 따라 달라진다.** 따라서 후보 순위와 cache hit, 이후 출력 경로는 바뀔 수 있다. 이는 전송량을 늘리는 변경은 아니지만 단순 이름 변경도 아니다. `full` 정규화의 고정 점수에서는 위치별 top-N으로 전체 top-N을 찾는 성질을 독립 계산과 대조한다.

## G=M 적용 범위

Config에는 phase별 검증 노드 수 하나만 저장한다. 엔진이 읽는 기존 `*_tree_max_nodes`와 `duet_tree_nv`는 그 값의 읽기 전용 별칭이다. 생성 중에 그 노드 예산을 적용하며, 생성한 노드는 hit 시 모두 검증한다. 현재 설정에 사후 점수 rerank를 추가하지 않는다.

G>M의 이전 연구 실행, root별 제한 없이 넓게 생성한 뒤 prefix M개만 내보내던 equal-budget 실험은 **ver1.0에서 재현**한다. 이 결과를 G=M의 새 실험 결과로 표시하면 안 된다. G=M은 탐색 중 예산 소진 시점과 자식 수를 바꾸므로 기존 AL 개선 수치를 그대로 승계하지 않는다.

탐색 버퍼는 초기화 때 `R + C×sum(round_widths)` 크기로 할당하고 CUDA graph에서 재사용한다. Root별 출력 버퍼는 M 크기이다. 이번 변경 때문에 inference 중 매번 메모리를 재할당하는 경로를 추가하지 않았다.

## 이전 생성 제한 완화 실험의 실제 tree 크기

자료: `equal_budget_20261014/RESULTS.json`이 선정한 두 서버의 12개 tree 실행. 모델은 LayerSkip-Llama2-7B + AMD-Llama-135m, **FP16**, B=1, T=0.7, Spec-Bench480문항560턴, K1=K2=8, phase별 R=W=8이다. C3은 생성 상한192, C8은512로 두어 해당 forward 예산에서 생성 상한이 걸리지 않게 했다. 최종 검증 상한 M은32 또는64였다.

`GENERATED_NODES.json`은 원시 forest 기록 **17,691,424개 root tree**를 다시 집계한 결과다. Root 토큰 자체는 노드 수에서 제외한다. 모든 exported forest를 포함하므로 warmup·마지막 step에서 준비한 작업도 포함되며, 실제 hit된 root만의 통계가 아니다. 표의 평균은 각 서버의 평균을 동일 가중한 값이다.

집계 JSON에는 phase·root 순위별 histogram과 원시 gzip 파일의 SHA256이 포함된다. 대용량 원시 trace는 Git에 추가하지 않았다. 재계산하려면 기존 `RESULTS.json`의 raw 경로와 `GENERATED_NODES.json`의 trace 경로에 해당 자료를 복원해야 한다.

| 방식 | 점수 1위 DS root의 평균 생성 노드 | 점수 1위 PS root의 평균 생성 노드 | root 하나의 실제 최대 노드 |
|---|---:|---:|---:|
| Last C3 / M32 | 44.72 | 70.04 | 156 |
| Frontier C3 / M32 | 39.07 | 63.39 | 156 |
| Last C8 / M64 | 141.08 | 229.01 | 456 |
| Frontier C8 / M64 | 120.65 | 210.27 | 456 |

8개 root 전체의 평균은 C3에서24개, C8에서64개이다. 상위 root에 계산이 집중되므로 1위 root의 평균은 더 크다. 각 phase 전체 생성량은64C개로 고정돼 있다. 최대값156/456은 DS와 PS, 두 서버 모두에서 관측됐다.

첫 round는 root8개에 한 행씩 배정된다. 한 root가 이후 최대 폭을 차지할 때 C3은 첫 확장 후 자식3개까지만 다음 입력이 될 수 있어 `3×(1+3+6×8)=156`, C8은 `8×(1+7×8)=456`이다. 이는 이번 K=W=R=8 조건의 상한이며 다른 설정의 보편적 상한은 아니다.

기존 보고서의 C3 hit tree 약23~28노드는 **M32로 자른 뒤 실제 hit 시 검증한 노드 수**이다. 위 표의 생성 크기와 다른 모집단·단계의 숫자다. 미선택 subtree의 target 검증 결과가 없어, 더 많은 노드를 보내면 AL이 얼마나 증가할지는 이 기록만으로 확정할 수 없다.

## 호환성과 다른 서버 이식

- `Config`의 dataclass 필드·직렬화·CLI에는 새 이름과 단일 M만 남긴다.
- 의미가 같은 옛 API 이름은 생성 시 경고와 함께 변환한다. `duet_p2_budget`→`duet_p2_forward_width`, `duet_tree_root_count`→`duet_p2_root_count`, 동일한 G/M→단일 M이다.
- 옛 scale, 수동 top-L, 0이 아닌 threshold, 충돌하는 G/M은 오류로 처리한다. 조용히 다른 알고리즘으로 실행하지 않는다.
- LLM 생성자가 알 수 없는 `duet_` 설정을 무시하던 동작을 수정했다. 잘못된 이름은 모델 로딩 전에 오류로 처리한다.
- `dataclasses.replace`를 통한 draft Config 재구성과 B1/B>1 scheduler의 KV 예약을 확인한다. DS 실행기와 scheduler는 동일한 입력 폭 계산 함수를 사용한다.
- 시작 로그와 runtime manifest에 실제 root 수, 입력 폭 상한, G=M, 자동 L/N을 기록한다. 각 실행기의 실제 round 폭은 `[tree-policy-resolved]`에서 확인한다.
- 기존 실험 JSON·원시 로그·완료 보고서는 덮어쓰지 않았다. `MLSys-ver1.0`과 다른 서버의 알고리즘 실험은 별도 기준으로 유지한다.

## 검증과 재현

**최종 39개 suite, 370개 테스트 통과. 실패 0, 오류 0, skip 0.** GPU는 RTX4090의 GPU2, Python3.12·Torch2.8.0+cu128이다. `source_unchanged=true`이고 검사 시작 시 runtime·bench·test 변경 사항이 없었다. CLI→실제 checkpoint Config 연결, 폐기된 옵션 거부, 구형·신형 로그 파싱도 통과했다.

최종 검사 결과는 `checks/CHECKS.json`, 소스 해시는 `checks/CHECKS_PROVENANCE.json`이다. 결과 집계·CLI 도움말·추가 검사 상태는 `VALIDATION.json`에 기록한다. 최종 suite별 로그는 `CHECK_LOGS.tar.gz`, 개발 중 실패·중간 실행 기록은 `DEVELOPMENT_CHECKS.tar.gz`에 보존한다.

개발 중에는 제거된 필드를 참조하던 fixture, scheduler 생성 fixture와 DS 실행기의 지역 변수 누락 등을 수정했다. 중간 전체 검사도 370개가 통과했으나 CLI/로그 코드를 검사 중 변경했으므로 최종 근거로 사용하지 않았다. 커밋 후 전체 검사를 다시 수행한 것이 위 최종 결과다.

```bash
CUDA_VISIBLE_DEVICES=2 SSD_CUDA_ARCH=8.9 SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp \
/home/chokwans99/PSD/ssd/.venv/bin/python results/mlsys_ver11/check_suites.py \
  --output-dir results/mlsys_ver11/recheck

python results/mlsys_ver11/analyze_generated_nodes.py
```

검사는 Config/CLI, GPU eager·graph parity, 온도 0과 확률적 경로, P1/P2, B>1, proposal·KV·집계를 포함한다. `check_cli.py`는 실제 7B/135M checkpoint의 **설정 파일만** 읽어 CLI→Config 연결과 구형·신형 로그의 설정 추출을 검사한다. 모델 가중치는 로딩하지 않는다.

실제 7B/70B 모델의 full 온라인 AL·TPS를 새 설정으로 재평가한 결과는 아니다. 특히 자동 L과 G=M의 정책 변화에 대한 품질 비교는 후속 실험에서 수행해야 한다. 다른 서버에서는 다음 순서로 확인한다.

1. 해당 GPU에서 회귀 검사를 재실행하고 `source_unchanged`와 오류·skip을 확인한다.
2. 기존 모델·입력·온도에서 새 API로 실행한다. 로그의 root 수, 실제 round 폭, L/N, G=M을 보관한다.
3. ver1.0에서 이미 G=M이었던 설정은 동일한 DS round 폭으로 맞춘다. 이 경우 우선 비교할 정책 차이는 자동 top-L이다.
4. 과거 G>M이었던 설정은 생성 상한 변화가 함께 있으므로, 그 AL과 새 결과를 같은 설정으로 표시하지 않는다.
