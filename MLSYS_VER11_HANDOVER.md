# MLSys ver1.1 작업 기준 및 다른 서버 인수인계

> **후속 작업 기준은 `MLSys-ver1.3`다.** [ver1.3 인수인계](MLSYS_VER13_HANDOVER.md)를 먼저 확인한다. 아래는 ver1.1 전달 당시의 기록이며, G=M·입력 폭·자동 top-L 등의 파라미터 계약은1.3에서도 유지한다.

2026-10-15 KST. 사용자 결정에 따라 **앞으로의 구현·파라미터 탐색은 `MLSys-ver1.1`을 기준으로 진행한다.** DS=P1, PS=P2이다. 다른 서버에서 진행 중인 알고리즘 변경은 이 기준 위에서 별도로 검토·병합한다.

## 받을 브랜치와 읽을 문서

| 항목 | 값 |
|---|---|
| 저장소 | `https://github.com/KwanyoungCho/PSD.git` |
| 작업 기준 브랜치 | **`MLSys-ver1.1`** |
| 부모 기준 | `MLSys-ver1.0@82d964e` |
| 구현 커밋 | `fe699aa5b88e7e5b8cb11ce04715af84e02f31c6` |
| 상세 보고서·검증 증거 커밋 | `14878f5` |
| 이 서버의 현재 worktree | `/home/chokwans99/PSD-MLSys-ver0.1` — 디렉터리 이름과 달리 실제 브랜치는1.1 |
| 상세 변경·파라미터 표·사용 예시 | [results/mlsys_ver11/REPORT.md](results/mlsys_ver11/REPORT.md) |
| 최종 검사 요약 | [VALIDATION.json](results/mlsys_ver11/VALIDATION.json) |
| 검사별 결과·소스 해시 | [CHECKS.json](results/mlsys_ver11/checks/CHECKS.json), [CHECKS_PROVENANCE.json](results/mlsys_ver11/checks/CHECKS_PROVENANCE.json) |
| 이전 tree 생성 크기 재집계 | [GENERATED_NODES.json](results/mlsys_ver11/GENERATED_NODES.json) |

이 문서는 위 두 커밋 이후의 전달 문서다. 수신자는 `git log -1`로 실제 수신 커밋을 기록한다. `VALIDATION.json`의 complete는 파라미터 구현·회귀 검증·기존 trace 재집계의 완료를 뜻한다. 새 설정의 full-model AL/TPS 평가나 모델별 최적 파라미터 탐색 완료를 뜻하지 않는다.

## 다른 서버에서 받는 방법

기존 PSD 저장소에서 다음을 실행한다. 기존 작업과 진행 중인 실험을 유지할 수 있도록 새 worktree를 사용한다. 동일한 경로·로컬 브랜치가 이미 있으면 새 이름을 정한다.

```bash
git fetch origin MLSys-ver1.1
git worktree add -b review/mlsys-ver1.1 ../PSD-MLSys-ver1.1-review origin/MLSys-ver1.1
cd ../PSD-MLSys-ver1.1-review
git log -1 --format='%H %s'
git status --short
git merge-base --is-ancestor fe699aa5b88e7e5b8cb11ce04715af84e02f31c6 HEAD
```

`merge-base` 종료 코드0은 구현 커밋을 포함한다는 뜻이다. 처음 받은 상태에서 `git diff fe699aa HEAD -- ssd`는 비어 있어야 한다. 이후 추가 구현이 게시됐다면 해당 diff도 검토한다. Sparse checkout을 사용하는 서버는 `ssd`와 `results/mlsys_ver11`이 포함되도록 범위를 설정한다.

새 실험은 이 worktree와 새 결과 디렉터리에서 실행한다. 진행 중인 실험의 runtime·manifest나 과거 원시 결과를 덮어쓰지 않는다. 모델 가중치, 데이터셋, 대용량 forest trace는 Git에 포함되지 않는다.

## 이번에 바뀐 설정

| 종전 설정 | ver1.1에서 사용할 설정·규칙 |
|---|---|
| `duet_p1_tree_forward_scale` | `duet_p1_forward_width`: DS 반복 forward 입력 수 상한 |
| `duet_p2_budget` | `duet_p2_forward_width`: PS 매 forward 입력 수 |
| `duet_tree_root_count` | `duet_p2_root_count`: DS 중복 제거 후 PS 전체 root 수 |
| phase별 생성 G와 검증 M | `duet_p1_tree_verify_nodes`, `duet_p2_tree_verify_nodes`만 지정. **G=M** |
| 수동 `duet_proxy_top_k` | 위치별 top-L을 전송 후보 수 N과 같게 자동 결정 |
| 네 가지 확장 threshold | 0으로 고정. 사용자 입력에서 제거 |
| 전송 여유분 | 최대 DS 중복 개수에서 자동 계산 |

DS는 첫 forward에서 모든 root R을 처리하고, 이후에는 `min(R, DS 입력 상한)`행을 사용한다. 입력 상한을 생략하면 R이다. PS는 root 수≤입력 수를 검사한다. DS 실행기와 scheduler의 KV 예약은 같은 입력 수 계산을 사용한다. 버퍼는 초기 할당 후 재사용한다.

의미가 같은 옛 Python API 이름은 경고와 함께 변환하지만, 새 실행기는 새 이름을 사용한다. CLI에서는 제거한 옵션을 받지 않는다. 옛 scale·수동 top-L·0이 아닌 threshold·G/M 불일치를 그대로 넣으면 오류로 중단한다. 잘못된 `duet_` 이름도 모델 로딩 전에 거부한다.

### 기존 결과와 같다고 가정하면 안 되는 부분

- **자동 top-L:** 70B preset은 위치별 추출량28→51, 전송 후보 수51 유지다. `topm` 정규화 분모가 달라져 root 순위·hit가 바뀔 수 있다.
- **G=M:** 과거 G>M이던 실험은 생성 도중 예산 소진 시점이 달라진다. 현재 구현은 생성 후 EAGLE 방식의 점수 rerank를 추가한 것이 아니다.
- **DS 입력 수:** 70B 시작 preset은 상한39로 기존 scale1.5의 실제 round 폭을 유지한다. 다른 모델에서는 기존 로그의 `round_widths`를 확인해 정한다. 입력 상한 생략이 옛 scale1.0과 항상 같지는 않다.
- **과거 tree 실험:** `equal_budget_20261014`의 G 제한 완화 실험은 ver1.0에서 재현한다. 당시 AL 수치를 새 G=M 결과로 옮겨 적지 않는다.

Root 점수 `e(1-q)`, 위치 혼합 `0.75 h_hat + 0.25 h_bar`, calibration 없는 draft 경로 점수, Last 기본 확장은 ver1.0에서 이어받았다. 이번 전달에서 다른 서버의 새 알고리즘을 추가 병합한 것은 아니다.

## 검증된 범위와 수신 서버 확인 순서

로컬 RTX4090 GPU2, Python3.12·Torch2.8.0+cu128에서 **39개 suite, 370개 테스트 통과; 실패·오류·skip 0개**다. 구현 커밋의 runtime·bench·test는 검사 전 clean이었고 `source_unchanged=true`다. P1/P2, B>1, greedy·확률적 경로, eager/CUDA graph parity, KV·proposal·집계가 포함된다. 이번 전달 문서 작성에서는 runtime을 추가 수정하지 않았다.

1. **회귀 검사 재실행.** Python 환경, 사용 가능한 GPU와 실제 GPU에 맞는 `SSD_CUDA_ARCH`를 수신 서버에서 설정한다. 4090은8.9, RTX PRO6000 Blackwell SM120은12.0이다. 출력 경로는 새로 정한다.

   ```bash
   export DUET_PY=/path/to/venv/bin/python
   export PYTHONPATH="$PWD/ssd:$PWD/ssd/tests"
   export PYTHONDONTWRITEBYTECODE=1
   # CUDA_VISIBLE_DEVICES와 SSD_CUDA_ARCH는 이 서버에 맞게 설정
   "$DUET_PY" -B results/mlsys_ver11/check_suites.py \
     --output-dir results/local_ver11_checks
   ```

   종료 코드, suite 건수, failures/errors/skipped, `source_unchanged`를 함께 확인한다. SM120의 알려진 SGL 미지원 skip은 통과 개수에 넣지 않고 따로 기록한다. 커밋된 `checks` 폴더를 새 결과로 덮어쓰지 않는다.

2. **모델별 짧은 기능 검사.** 기존 모델·dtype·온도·입력 형식을 유지하고 새 API로 연결한다. DS/PS root 수, G=M, 자동 L/N, 실제 `round_widths`, runtime manifest를 저장한다. `check_cli.py`의 로컬 모델 경로는 참고용이며, 이 검사는 가중치 로딩 없는 Config 연결 검사다.

3. **full dataset 비교.** 새 설정으로 AL, P1/P2/miss별 AL, cache hit를 재측정한다. 원본과 비교할 때 생성 예산과 실제 forward 입력 수를 함께 기록한다. 이미 G=M이던 설정에서는 자동 top-L의 영향을 우선 분리한다. G>M→G=M 변경은 별도 차이로 표시한다.

4. **모델별 파라미터 탐색.** 70B preset은 시작점이다. Exit56, K1/K2=8/4, DS/PS 입력 상한39/15, DS 위치당 root3·PS root10, 노드 상한12/8, C3을 모든 모델의 최적값으로 취급하지 않는다. 다른 모델에는 layer 수와 phase별 시간 예산에 맞는 설정을 찾는다.

최종 개별 검사 로그는 [CHECK_LOGS.tar.gz](results/mlsys_ver11/CHECK_LOGS.tar.gz), 개발 중 중간 기록은 [DEVELOPMENT_CHECKS.tar.gz](results/mlsys_ver11/DEVELOPMENT_CHECKS.tar.gz)에 있다. 원시 forest 재집계의 최대 생성 크기는 C3에서156, C8에서456개이며 root 토큰 제외·검증 상한 적용 전 통계다. 이는 새 runtime의 성능 평가가 아니다.

## 다른 서버 알고리즘을 합칠 때

수신한1.1 기준에서 별도 작업 브랜치를 만들고 알고리즘 변경만 검토한다. 기존 서버의 launcher·Config·scheduler를 통째로 덮어쓰면 제거된 파라미터와 G>M 설정이 다시 들어올 수 있다. G=M·후보 추출량 자동화·DS/PS 입력 수 계약을 유지하고, 의도적으로 바꾸는 알고리즘과 그 실험 조건을 따로 기록한다.

최종 리뷰에는 구현 diff, 실제 적용 파라미터, 기능 검사, full dataset의 유효 문항·턴 수, phase별 AL/hit, 필요한 component latency를 함께 제출한다. 기존370개 회귀 통과를 새 알고리즘의 AL/TPS 개선 증거로 대체하지 않는다.
