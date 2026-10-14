# MLSys-ver0.2: 리뷰 수정 병합과 tree 구조 실험 인수인계

작성: 2026-10-14. 대상 브랜치: **`MLSys-ver0.2`**. 이 문서는 이번 push를 검토하는 작업자의 시작점이다.

이번에 요청한 실험 결과는 **동일 생성 예산에서 chain 모양·마지막 깊이 경쟁·전체 말단 경쟁을 비교한 tree 구조 실험 14개**다. 다른 서버에서 진행한 연구 결과를 대신 병합하거나, 해당 서버의 launcher/manifest를 덮어쓴 작업은 아니다.

## 1. 커밋과 출처를 먼저 구분

| 커밋 | 내용 | 검토 방법 |
|---|---|---|
| `76c98f9` | 이번 통합 전 원격 ver0.2 기준 | 이전 release 기준 |
| `091fa64` | q-only 기본값, 동일 예산 tree 실험 adapter·입력·회귀 검사 보존 | **14개 full 실험 당시의 소스 정본**. SOURCE_SHA256의 200개 파일과 모두 일치 |
| `fce56f1` | 14개 실험의 결과·문항별 CSV·그림·분석 스크립트·과거 실행 기록 | full 실험 결과 보존 커밋 |
| `8782668` / `8c06ef4` | 별도 리뷰 브랜치의 Blackwell 공통 수정 / 검증 자료 | `blackwell_review_20261014/REPORT.md` |
| `48383b1` | 위 두 작업의 실제 merge 및 의미상 충돌 해결 | 두 브랜치 병합 커밋 |
| `f6494f6` | CPU 참조 rollout의 정책 기본값도 `q_path`로 일치시킴 | **최종 병합 runtime 검증 기준** |
| `6fcbc7d` | Full tree 실험 문서의 BF16 오표기를 FP16으로 정정 | 결과 수치 변경 없이 근거와 이전 manifest 보존 |
| 이 문서를 추가한 후속 커밋 | 병합본 회귀·실제 모델 검사와 이식 지침 | `merge_validation_20261014/` |

`091fa64`는 실험을 새로 실행한 커밋이 아니라, 실행 당시 고정해 둔 작업 트리를 사후 커밋으로 보존한 것이다. SHA256 대조로 동일성을 확인했다. 기존 원시 JSON에 기록된 당시 HEAD/dirty를 수정하지 않았다.

문서의 **BF16 표기는 FP16으로 정정했다.** 양 서버 target config 해시와 frozen runtime의 dtype 선택을 대조한 결과다. 수치·원시 결과는 그대로이며 `DTYPE_AUDIT.json`에 근거, `fce56f1`과 `FINAL_ARTIFACT_SHA256_before_dtype_correction.json`에 정정 전 이력을 보존한다.

**Full AL 결과는 `091fa64` 소스로 얻은 결과다.** 병합본의 near-tie 정규화 수정 때문에 기존 후보의 동률 부근 순서는 달라질 수 있다. 기존 14개 결과를 `48383b1` 이후 runtime의 재실행 결과라고 해석하면 안 된다. 병합본에는 별도의 기능·회귀 검증을 수행한다.

## 2. 이번 병합에서 확인한 내용

| 항목 | 반영 내용 |
|---|---|
| 불필요한 토큰 디버그 출력 | `SSD_DEBUG_TOKENS=1`에서만 수행. 기본은 끔. 주요 설정·문맥 검사는 Python `-O`에서도 유지 |
| F/DS와 split 예산 | 명시 P2 예산이 있는 split 설정은 DS=F 허용. F−DS에서 예산을 유도하는 경우의 제약은 유지 |
| Target/draft config | draft config 재구성 시 target의 exit 범위를 보존. 모델별 layer/head/dtype 계약 확인 |
| Attention/workspace | cached-prefill의 auto/FI/SGL 경로별 검사. batch target/glue 256 MiB, draft P1/P2 128 MiB 기본값과 상세 실패 진단 |
| Dense/flat 수치 일관성 | vocabulary 합계 reduction을 같은 layout에서 계산하고 작은 순위표 배치를 맞춤. 후보 ID/순서와 출력/RNG를 별도 검사 |
| Worker 정리 | 초기화 실패 시 자신이 만든 worker만 제한 시간 내 정리. target/draft worker 사망 감시 |
| 실행 상태·환경 | queue 상태 helper, SentencePiece 의존성, ABI/모델/head/정책 runtime manifest |
| 이번 연구의 q-only 요청 | **calibration 없는 `q_path` 기본값 유지**. Config, runtime manifest, executor 로그 및 CPU 참조 rollout 허용 조건을 일치시킴 |
| 병합 회귀 | 두 브랜치의 검사 목록을 합침. 실제 executor 속성을 갖춘 연구 fixture로 교체하고 기본 정책 로그 검사 추가 |

기존 리뷰 보고서의 “calibrated 기본값 유지”는 **리뷰 브랜치 당시의 사실**이다. 이번 통합 기본값은 사용자가 이후 요청한 `q_path`다. calibrated 모드는 과거 결과 재현을 위해 명시적으로 선택할 수 있으나 기본으로 사용하지 않는다.

`q_path`는 점수 계산 방식이며, 동일 예산 실험의 모든 설정을 자동으로 켜는 이름은 아니다. R=W=8, K1=K2=8, Last/Frontier 경쟁, 비구속 생성 상한은 `mlsys_equal_budget.py`가 불러오는 **실험 adapter**에 있다. 일반 runtime에 Frontier/C3/M32를 새 기본값으로 강제하지 않았다. Miss 처리와 root 점수의 새로운 변경도 이번 merge에서 추가하지 않았다.

## 3. Tree 실험 결과를 읽는 순서

1. [최종 보고서](equal_budget_20261014/FINAL_REPORT.md): 수식, 예산, 조건부 AL, 깊이 제한, 해석의 한계.
2. [전체 그림 PDF](equal_budget_20261014/figs/ALL_FIGURES.pdf): 6개 그림을 한 번에 확인.
3. [서버별 전체 표](equal_budget_20261014/FINAL_TABLE.csv), [문항별 지표](equal_budget_20261014/QUESTION_METRICS.csv).
4. [통계·원시 파일 해시](equal_budget_20261014/FINAL_ANALYSIS.json), [모든 arm 세부 집계](equal_budget_20261014/RESULTS.json).
5. [논문 입력·출력 프로토콜](equal_budget_20261014/PAPER_PROTOCOL.json), [고정 입력](equal_budget_20261014/SPEC_BENCH_FULL.json), SOURCE_SHA256/MODEL_SHA256.

실제 조건: LayerSkip-Llama2-7B + AMD-Llama-135m full FP16, B=1, T=0.7, exit21, Spec-Bench 480문항/560턴, 출력 상한1024/문맥2048, 입력 절단 없음. 두 서버/seed에서 각각 7개 설정을 실행했다. 각 phase의 root 수와 forward 폭은 8, K1=K2=8로 활성 forward 입력64행씩을 맞췄다. 실제 검증 node 수와 전체 latency까지 같다는 뜻은 아니다.

| 방법 | 전체 AL | P1-hit AL | P2-hit AL | Miss AL | Hit (%) |
|---|---:|---:|---:|---:|---:|
| Chain 모양 C1/M32 | 3.437 | 4.033 | 2.899 | 2.494 | 81.88 |
| 마지막 깊이 C3/M32 | 3.640 | 4.492 | 3.542 | 2.605 | 77.45 |
| 전체 말단 C3/M32 | 3.693 | 4.528 | 3.586 | 2.617 | 77.42 |
| 마지막 깊이 C8/M32 | 2.904 | 3.315 | 2.718 | 2.601 | 80.78 |
| 전체 말단 C8/M32 | 2.925 | 3.345 | 2.744 | 2.599 | 79.95 |
| 마지막 깊이 C8/M64 | 3.610 | 4.586 | 3.377 | 2.628 | 76.83 |
| 전체 말단 C8/M64 | 3.694 | 4.661 | 3.440 | 2.690 | 77.47 |

- C3/M32의 AL 증가는 Chain 대비 Last **+5.91%**, Frontier **+7.44%**. Frontier 대 Last 차이의 95% 구간은 0을 포함하므로 둘의 우열은 확정되지 않았다.
- C8/M32는 생성 순서 prefix로 검증 node를 자를 때 깊이가 최대4로 제한된다. M64로 늘리면 회복하지만 C3/M32보다 추가로 좋다는 근거는 없다.
- P1/P2/miss AL은 각 source가 실제로 발생한 문항의 조건부 평균이다. 조건부 분모가 서로 다르며 miss AL 차이를 miss 알고리즘 개선으로 해석하지 않는다.
- 480개 문항을 두 번 실행했으므로 독립 문항 960개로 계산하지 않는다. 문항별로 반복을 평균한 뒤 paired bootstrap한다.
- 사용자 표현의 **q-global은 마지막 깊이 Last**다. Frontier는 이전 깊이의 미확장 말단까지 경쟁한다.
- root 가중치의 단독 효과, C3 주경로 보호 대조군, score rerank의 이득, B>1 AL은 이번 14개 실험으로 결론 내리지 않는다.

## 4. 다른 서버에서 병합 확인

현재 서버에서 진행 중인 연구 소스는 그대로 두고 별도 checkout에서 먼저 확인한다.

```bash
git fetch origin MLSys-ver0.2
git worktree add --detach ../PSD-ver02-review origin/MLSys-ver0.2
cd ../PSD-ver02-review
git log --graph --oneline -10
git merge-base --is-ancestor 8c06ef4 HEAD
git merge-base --is-ancestor 091fa64 HEAD
git show --remerge-diff 48383b1 -- ssd/ssd/config.py ssd/ssd/utils/runtime_manifest.py ssd/tests
```

두 ancestry 명령의 종료 코드가 모두0이면 리뷰 수정과 실험 코드가 포함된 것이다. `git merge-base --is-ancestor f6494f6 HEAD`도 확인한다. 필요 시 `git diff 76c98f9 f6494f6 -- ssd/ssd ssd/pyproject.toml ssd/uv.lock`로 이전 ver0.2와 runtime 차이를 본다. 다른 서버 연구 브랜치에 실제 merge할 때는 root/위치/tree 알고리즘 충돌을 확인하고 그 서버의 launcher·모델 경로·manifest를 유지한다. 강제 reset이나 서버별 파일 덮어쓰기는 필요 없다.

## 5. GPU 없이 결과 확인

NumPy가 설치된 Python으로 repository root에서 실행한다. shallow clone은 먼저 위 커밋 이력을 가져와야 한다.

```bash
python results/mlsys_ver02/merge_validation_20261014/verify_tree_results.py
python -B -m unittest discover -s results/mlsys_ver02/equal_budget_20261014 -p 'test*analysis.py' -v
```

첫 명령은 최종24개 산출물 해시, frozen commit의200개 소스, 6,720개 문항 행, 14개 arm의 평균·조건부 분모, 15개 paired 비교와 문맥 보호 종료 민감도 분석을 재검산한다. 새 merged checkout이 과거 소스와 같다고 요구하지 않고, **git의 `091fa64`와 비교**한다.

실험 폴더의 `checks/`, `protocol_checks/`, `OBSERVER_ERROR.json`, `OBSERVER_PID.json`, `superseded_plans/`는 당시의 이력이다. 과거 disk/external-GPU 실패와 폐기 계획도 남겼다. 현재 완료 판정은 `FINAL_ANALYSIS.json: state=complete`, `RESULTS.json`의 두 서버 complete, 본 병합 검증 기록을 기준으로 본다. 과거 observer PID로 프로세스를 종료하거나 폐기 계획을 재실행하지 않는다.

## 6. 원시 파일과 재현

Git에 포함: 결과 요약·문항별 CSV·입력 ID·그림·전체 분석 코드·실행 계획·소스/모델/원시 파일 해시. **약4.1 GB의 원시 output/forest·실패 attempt와 모델 가중치는 Git에 넣지 않았다.** 원시 자료는 이번 서버(hostname `orion`)의 `/data/chokwans99/equal-tree-paper-full-20261014/`에 보존했고 eslab18의 유효 실행도 그 안의 `remote_mirror/`로 모았다.

원시 자료까지 필요한 작업자는 해당 디렉터리를 디렉터리 구조 그대로 별도 전송하고 다음으로14개 유효 run JSON 해시를 확인한다.

```bash
python results/mlsys_ver02/merge_validation_20261014/verify_tree_results.py \
  --raw-root /path/to/equal-tree-paper-full-20261014
```

`FINAL_ANALYSIS.json.raw_manifest`가 최종 선택된14개 JSON의 정본 목록이다. 문항 CSV 재검산만으로 step 원시 기록의 독립 재집계를 대체하지 않는다. 전체 step/forest 재분석에는 `analyze.py`와 `final_analysis.py`를 사용한다. 후자는 이 서버의 BASE 경로가 고정되어 있으므로 복사본에서 새 raw 경로에 맞춰 수정하거나 같은 경로를 준비한다. 원본 결과를 덮어쓰지 않도록 별도 결과 디렉터리에서 실행한다.

Full 실험 재현은 `091fa64`의 별도 worktree에서 시작한다. 입력·계획은 `fce56f1`의 experiment 폴더를 사용하고, 계획의 모델/checkout/입력 경로 및 CPU/GPU 배정만 서버에 맞춰 바꾼다. 실제 채택 arm과 seed는 `FINAL_ANALYSIS.json.raw_manifest`, 실행 인자는 `RESULTS.json[server].arms[arm].parameters`가 기준이다. 새 merged runtime으로 같은 실험을 하면 새 검증 attempt로 기록하며 기존 결과와 섞지 않는다.

## 7. 병합 후 검증 및 남은 범위

병합 검증은 완료했다: **33 suite/312개 검사 통과, 실제 모델6/6 실행 통과, greedy AR 일치8/8, Last/Frontier adapter2/2 통과 및 phase forest294개의 동일 예산 확인**. 상세 결과·소스 해시·실행 명령은 [병합 검증 보고서](merge_validation_20261014/REPORT.md)를 참조한다. 과거 review의302개 검사 결과와 이번 병합본 결과는 서로 다른 실행이다.

다른 서버에 남는 검증은 실제 Blackwell/70B의 B1·B2·B8 workspace 및 실행, 서버 전용 adapter의 F 자동 증가 제거, report 상태 helper 연결, 그 서버에서 개발한 알고리즘과의 통합이다. 이번 push로 이 원격 작업들까지 완료했다고 표시하지 않는다. 사용자 요청에 따라 이번 tree 연구 품질 비교는 B=1이며 B>1 AL은 후속 범위다.
