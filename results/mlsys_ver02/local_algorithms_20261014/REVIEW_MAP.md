# 다른 서버에서의 검토 순서

이번 범위는 **우리 서버에서 이미 연구한 알고리즘을 ver0.2에 이식**하는 작업이다. 다른 서버의 진행 중인 tree/root 알고리즘은 가져오지 않았다.

## 커밋과 비교 기준

- Before: `MLSys-ver0.2@18e912f` (공통 수정·실행 최적화 반영 후, 연구 점수 반영 전).
- Runtime: `cd2506c`.
- 테스트·ablation harness: `992b8b8`.
- 측정 중 위 runtime/bench 파일을 변경하지 않는다. 최종 결과·분석 자료는 후속 문서 커밋이다.
- 논문 원본 `a82f7d2` 대비 실행 최적화 성능은 이전 보고서의 별도 비교다. 이번 A/B를 원본 논문 재현이라고 부르지 않는다.

## 파일별 변경

| 내용 | 주 파일 | 검토할 계약 |
|---|---|---|
| Root 토큰·위치·full 정규화 | `ssd/ssd/engine/helpers/root_policy.py` | 실제 proposal/correction law가 아닌 cache 점수 |
| B>1 dense root | `batch_tree_sampling.py` | exact verifier ladder 유지, candidates만 공유 helper |
| B>1 flat root | `flat_tree_sampling.py` | 동일 terminal 재귀와 score, 실제 verifier는 `exact=True` |
| B1 eager/graph 및 chain | `p2_tree.py`, `verifier.py` | 동일 온도/sampler_x/valid_k, actual bonus 위치 |
| 그래프 준비 연결 | `model_runner.py`, `batch_tree_verify.py`, `batch_tree_warmup.py` | 실행과 prewarm의 정책·깊이·형제 폭 일치 |
| Reach 보정·frontier 설정 | `tree_expansion_policy.py` | q 대체 금지, 보정표 hash/범위/fallback 명시 |
| Round별 정수 배분 | `tree_gain_allocation.py` | request/root별 예산 독립, future reserve, GPU capture 가능 |
| P1/P2 tree 실행 | `p2_tree_executor.py`, `p2_tree.py`, `draft_runner.py` | 실제 parent 깊이의 RoPE, 미확장 node만 선택, eager/graph 동일 |
| 보정표 배포 | `tree_calibration/*.json`, `ssd/pyproject.toml` | wheel 안 JSON hash가 원본과 동일 |
| 정책 계약 로그 | `config.py` | 운영 기본값 하나, 지원하지 않는 과거 raw/top-M 경로 명시 오류 |
| 독립 수식·경계 검사 | `ssd/tests/test_local_research_policy.py` | FP64 scalar reference, raw q, RNG/input 불변, eager/graph |
| 분리 비교 | `ssd/bench/mlsys_local_algorithms.py` | benchmark process에서만 정책 교체, 검증/샘플링 미변경 |

## 재현

```bash
git diff 18e912f..cd2506c -- ssd/ssd ssd/pyproject.toml
git show 992b8b8
CUDA_VISIBLE_DEVICES=7 SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp \
PYTHONPATH=ssd OMP_NUM_THREADS=2 <python> -B \
results/mlsys_ver01/extensions/check_suites.py --output-dir <새 검사 폴더>
<python> -B ssd/bench/mlsys_tree_campaign.py \
  --plan <서버 경로를 바꾼 l2_FULL_PLAN.json 사본> \
  --directory <새 결과 폴더> --gpus <비어 있는 target,draft GPU> --port <빈 포트>
```

결과를 다시 분석할 때 RAW_L2/RAW_L3/RAW_L3_RECHECK archive를 풀고 `analyze.py`의 primary·retry raw 경로를 복구 위치에 맞춘다. 실패한 원래 Llama3 proxy 실행은 RAW_L3에 남아 있고 정상 retry만 분석에서 선택한다. JSON 안 원래 측정 명령/경로/commit/hash는 덮어쓰지 않는다. `make_figs.py`는 14개 primary arm이 완료돼야 실행된다.

## 이후 병합 시 충돌 주의

1. Root 정책을 합치면 B1, B>1 dense, flat 세 경로 모두 같은 수식을 써야 한다. 새 서버의 수식 파일 하나만 복사하지 않는다.
2. 다른 calibration으로 교체하면 source prompt, dtype, phase/sibling/q-bin 의미와 C 범위를 함께 확인한다. 온라인 평가 결과로 fitting한 뒤 같은 평가를 held-out이라고 쓰지 않는다.
3. 다른 tree selector를 합치더라도 실제 q와 생성 순서 prefix를 유지한다. 사후 재정렬 후 원래 q로 검증하지 않는다.
4. 현재 Miss JIT chain, mixed-miss 미채택, target packed 미채택, stream 명시 설정은 이번 연구 점수 반영과 별개다.
5. 외부 서버의 launcher·model 경로·manifest를 그대로 덮어쓰지 않는다. 이번 자료에 GPU 장애/성능 회귀로 잘못 분류하지 않도록 외부 process의 초기화 간섭 기록도 남겼다.
6. 같은 seed 출력의 변경은 정책 변경 시 가능하다. 독립 verifier/proposal 검증을 유지하고, 결과 수치만 보고 lossless 여부를 판단하지 않는다.

최종 채택 해석과 전체 수치는 REPORT.md/TABLES.md를 기준으로 읽는다. 오래된 ver0.2 보고서에서 “연구 알고리즘 제외”라고 한 것은 `18e912f`까지의 역사적 범위다.

## 자료 확인 순서

1. REPORT.md의 2×2 결론과 비용·한계를 먼저 읽는다. TABLES.md에는 전체 7개 arm과 CI가 있다.
2. FROZEN_SOURCE.json / CHECKS_PROVENANCE.json / FINAL_AUDIT.json으로 검사한 소스와 실험 소스가 같은지 확인한다.
3. ROOT_COMPONENT.json은 동일 입력 점수 계산 비용이고, PROFILE_RESULTS.json은 알고리즘에 따라 방문 문맥이 다른 online 구간 관측이다. 두 결과의 목적을 혼동하지 않는다.
4. ARCHIVES.json의 SHA-256으로 원시 archive를 확인하고, 원시 campaign에서 종료 상태·외부 process·명령을 확인한다. 실패 archive도 지우지 않았다.
5. 알고리즘별 출력이 다를 수 있으므로 output equality를 새 정책의 lossless 판정으로 사용하지 않는다. 실제 q와 correction 법칙의 코드·독립 검사로 확인한다.

Profile 후처리는 B>1 draft의 `batch_kv_restore`를 매 step anchor로 사용해 generate 호출별 epoch를 구분한다. 기존 plotter의 `draft_recv_request` anchor는 이 batched 경로에 없으므로 그대로 호출하면 epoch를 잘못 나눈다. 이번 수정은 분석 script 안의 변환이며 엔진 측정 코드 변경이 아니다. 각 구간의 포함 step 수와 전체 generate epoch 수가 원시 요청 기록과 같은지도 검사한다.
