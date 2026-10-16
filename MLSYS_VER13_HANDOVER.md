# MLSys ver1.3 인수인계 — ver1.2 이후 전체 변경

작성: 2026-10-17 KST. **이후 이 서버의 작업 기준 브랜치는 `MLSys-ver1.3`이다.**
저장소: `https://github.com/KwanyoungCho/PSD.git`.
새 작업 폴더: `/home/chokwans99/PSD-MLSys-ver1.3`.
기존 `/home/chokwans99/PSD-MLSys-ver0.1`의 `MLSys-ver1.2`와 미커밋 작업은 보존했다.
연구 브랜치 이름과 `ssd/pyproject.toml`의 upstream 패키지 버전 `0.2.0`은 별개다.

## 1. 이번 반영의 범위와 비교 기준

| 기준 | 커밋 | 의미 |
|---|---|---|
| 원격에 전달된 ver1.2 | `899e4eb1` | `origin/MLSys-ver1.2`의 기존 전달 기준 |
| 이번 실험의 before | `1a2e349` | 위 커밋에 당시 ver1.2의 미커밋 집계·진단 코드를 더한 snapshot |
| 검증된 결합 runtime | `51030a0` | 응답 snapshot + 검증 입력 재사용 + draft KV 복원 순서 변경 |
| 실험 결과를 묶은 커밋 | `b272892` | 앞 runtime을 바꾸지 않고 원시 결과·분석·보고서를 기록 |
| 이번 전달 | `MLSys-ver1.3` | 위 변경과 남아 있던 timeline 도구·검사·과거 진단 자료를 함께 전달 |

**성능표의 before는 `899e4eb1` 자체가 아니라 `1a2e349`다.**
따라서 이전에 승인된 집계 정책 변경을 이번 최적화의 성능 이득으로 설명하지 않는다.
최종 runtime의 93개 Python 파일은 `51030a0` 및 원격의 결합 검증 소스와 byte 단위로 같다.
[RUNTIME_MANIFEST.json](results/mlsys_ver13/RUNTIME_MANIFEST.json)에 전체 SHA256을 기록했다.
변경 경로 전체 목록은 [CHANGED_FILES.json](results/mlsys_ver13/CHANGED_FILES.json)이다.

후보 점수·위치 점수·draft 확률·수락 법칙·tree 확장 및 선택 알고리즘·G=M 계약·모델 preset과
사용자 파라미터는 이번에 바꾸지 않았다. 새 성능 옵션도 없다. B=1은 **활성 요청 1개**라는 뜻이다.
TP2는 **하나의 target 모델을 GPU 두 장에 나누어 실행**한다는 뜻이며 batch=2가 아니다.

## 2. 채택한 실행 최적화: 기존 동작 → 변경 동작

### A. B=1 응답 메타데이터를 한 번만 CPU로 읽고 공유

- 기존: 응답 후 `valid_k`, source, tokens, tree topology를 서로 다른 소비자가 GPU에서
  반복해서 scalar/list로 읽었다. 같은 응답에 대한 동기화와 Python 변환이 겹쳤다.
- 변경: 기존 fused int64 packet을 한 번 읽어 immutable `HostResponse`를 만든다.
  Sequence 갱신, 검증 길이·tree routing, topology 검사, 지표 기록이 이 snapshot을 공유한다.
  GPU에 있는 토큰·logit도 그대로 보존한다. `tolist()`와 CPU 처리를 전부 없앤 것은 아니다.
- 적용: 실제 B=1 응답 경로. B>1은 기존 처리 유지. Packet layout·전송 개수·RNG는 동일하다.
- 파일: `helpers/response_snapshot.py` 신규, `helpers/speculate_types.py`,
  `speculator_async.py`, `verifier.py` (`ssd/ssd/engine/` 아래).
- 검사: packet 변경과 snapshot이 alias하지 않음, 잘못된 크기·valid_k 거부,
  ragged/uniform sequence 확장 결과가 기존과 동일함.

### B. B=1 target 검증 입력을 재구축하지 않고 재사용

- 기존: 매 step CPU token list → 새 pinned tensor → GPU 복사 및
  positions/slot/context/page/cu tensor 생성이 있었다. 이미 GPU에 받은 토큰도 다시 구성했다.
- 변경: rank0는 수신한 GPU 토큰을 직접 사용한다. positions, slot, context length, page table,
  cu buffer는 runner 수명 동안 유지한다. Page table은 CPU block tuple이 바뀔 때만 복사한다.
  작은 Triton kernel 하나가 현재 position·실제 물리 KV slot·length/cu를 채운다.
- Target TP follower는 기존 공유 메타데이터를 유지하면서 두 개의 고정 pinned staging slot을
  번갈아 쓴다. CUDA event로 복사가 끝나기 전 host slot을 덮어쓰지 않게 한다.
- 모든 허용 query 행 수를 초기화 때 준비하고, prefix/context 값으로 새 JIT가 생기지 않게 한다.
  최초 생성·컴파일 비용 자체가 없어지는 것은 아니므로 inference 측정 전 warmup이 필요하다.
- 기존 CPU 문맥 길이를 재사용하여 tree 실행의 GPU scalar read 하나를 줄이고,
  `last_only=False` 검증에서 사용하지 않는 sample-temperature 준비도 생략한다.
- 적용: target의 실제 B=1 DUET 검증, non-EAGLE. Chain과 tree 모두 사용한다.
  일반 B>1 입력은 기존 경로다. B>1 실행의 tail이 B=1이면 이 조건에 들어갈 수 있다.
- 파일: `helpers/single_verify_inputs.py` 신규, `model_runner.py`, `verifier.py`.
- 검사: 여러 길이·K, 비연속 physical page, GPU token/host staging 입력, 반복 재사용,
  잘못된 prefix와 미할당 page를 확인했다. 논리 query 수·dtype·KV 주소는 기존과 같다.

### C. Cache-hit 응답을 이전 draft tree KV 복원보다 먼저 전송

- 기존 순서: 요청 수신 → 이전 tree의 수락 경로 KV 복원 → hit 조회/응답 → glue·DS·PS.
- 변경 순서(hit): 요청 수신 → hit 응답 → 이전 수락 경로 KV 복원 → glue·DS·PS.
- 변경 순서(miss): 요청 수신 → **KV 복원 → JIT draft** → 응답 → 후속 phase.
  Cache가 비어 있는 최초 JIT도 같은 순서를 지킨다.
- 근거: cached hit의 tokens/logits를 보내는 일은 canonical prefix KV를 읽지 않는다.
  Target이 기다릴 필요가 없는 복원을 뒤로 보내 target 실행과 겹칠 기회를 만든다.
  복사량이나 draft forward 수를 줄인 변경은 아니다.
- 새 hit를 서비스하면서 이전 staging/topology가 교체될 수 있으므로 이전 상태를 별도로
  보관하고 한 번만 복원한다. Sequence identity, EOS/출력 상한으로 절단된 실제 수락 prefix,
  fragmented KV page의 목적 주소는 기존 규칙을 보존한다.
- 파일: `draft_runner.py`의 `_restore_previous_tree_kv`, `_service_spec_request`,
  `hit_cache_and_respond` 내부 callback. 사용자 설정은 추가하지 않았다.
- 기존 KV 정확성 오류의 수정이나 새 prefix-cache 정책 도입이 아니다.
  Chain에는 이전 tree KV 복원 자체가 없어 이 변경의 직접 효과가 없다.

A와 B는 함께 적용하며 C도 같은 runtime에 들어 있다. CPU/GPU 실행기 중 하나를 고르는
새 옵션을 만들지 않았다. A 단독의 미세한 개선을 확정적으로 주장하지 않는다.

## 3. 성능 근거와 한계

원시·분석·정확성 기록은
[target_gap_optimization_20261017/REPORT.md](results/mlsys_ver12/target_gap_optimization_20261017/REPORT.md),
[비교 그림](results/mlsys_ver12/target_gap_optimization_20261017/LATENCY_COMPARISON.png),
[source별 CSV](results/mlsys_ver12/target_gap_optimization_20261017/SOURCE_LATENCIES.csv)에 있다.

Profile OFF, 같은 GPU·입력·출력·수락 사건의 step끼리 비교했다. 두 process 반복과
12 UID cluster bootstrap을 공개했다. 모든 값은 **전체 decode step latency**다.

| 독립 비교 | Target + llama-68m | 전→후(ms) | 감소 | 감소율 95% CI |
|---|---|---:|---:|---:|
| A+B 대 before | Llama2-7B, tree | 32.365→31.947 | 1.29% | [0.55, 1.96]% |
| A+B 대 before | Llama2-13B, tree | 45.213→44.534 | 1.50% | [1.20, 1.81]% |
| A+B 대 before | Llama2-7B, chain K10/40 roots | 24.749→24.029 | 2.91% | [2.70, 3.14]% |
| C 단독 대 before | Llama2-7B, tree | 31.882→31.451 | 1.35% | [1.00, 1.72]% |
| C 단독 대 before | Llama2-13B, tree | 44.513→44.255 | 0.58% | [0.08, 1.00]% |

마지막 두 행은 **C 단독 비교**이며 A+B와 실행 캠페인이 다르다. 비율을 합산하지 않는다.
13B C의 두 반복은 +1.11%와 +0.05%로 두 번째는 거의 중립이다.
Miss만 보면 C의 step은 7B 약0.10%, 13B 약0.33% 느렸고 hit에서 이득을 얻었다.
Process 변동까지 bootstrap CI가 모두 대표하는 것은 아니다.

구간을 직접 본 warmed profile에서 C는 target 요청 send 종료→응답 완료를
7B 3.552→2.577ms, 13B 3.206→2.692ms로 줄였다.
이전 final logits 종료→다음 target forward 시작 공백은 각각7.559→6.385ms,
7.761→7.199ms였다. 이 값은 별도 profile ON 진단이며 위 비계측 절감량과 합치지 않는다.
순수 NCCL 통신 시간이나 Transformer kernel 개선으로 해석하지 않는다.

결합 A+B+C는 GPU 회귀52개와 모델별12입력 단일-process 실행을 통과했다.
7B 출력12건/수락 event918개, 13B 출력12건/event979개가 기준과 정확히 같다.
**결합 전체의 반복 A/B는 아직 하지 않았다.** 단일 실행 step31.038/43.745ms를 결합 개선율로
사용하지 않는다. 개별 채택 근거와 결합 정확성 검증을 구분하여 반영한 릴리스다.

측정 조건:

- eslab18/nova RTX3090, BF16, T0.7, seed9101, B1. 로컬 orion의 절대 수치와 섞지 않는다.
  이번 원격 성능 진단은 사용자의 별도 지시로 실행했다.
- 7B target GPU2/draft3. 13B target GPU4·5(TP2)/draft6. PIX topology, CPU affinity 분리.
- Tree K1=K2=3, C3, G=M40, DS 위치당 root3, PS root9.
  DS 후속 round 입력 상한9, PS 입력9이며 DS 첫 round는 실제 root 수에 따라 다르다.
- Exit21/26(0-based), context 상한2048, 입력 cap512, 출력 cap192, warm3.
  사전 고정 Spec-Bench12턴(6과제)×2회/process. Full 480문항 품질 평가가 아니다.
- Chain은7B, K1=K2=10, phase당40 roots의 별도 조건이다.
- 완료36실행과 준비 단계 중단4실행을 구분한다. 소스 hash 실행 전후 동일,
  관측 최대66°C, 할당 GPU당 동시 process 1개 이하. 초기 cold-JIT profile은 최종 이득에서 제외했다.
- B>1, T=0 full-model, 70B/Blackwell, 13B chain의 새 성능 검증은 하지 않았다.
  기존 지원을 삭제하지 않았으나 이 수치를 해당 조건으로 일반화하지 않는다.

## 4. committed ver1.2 대비 함께 전달하는 기존 승인 사항

아래는 새 latency 최적화와 구분한다. 이전 작업 폴더에 남아 있던 변경이며
실험 before에도 집계 코드는 동일하게 들어 있다.

### D. Profile OFF에서 비정상적으로 긴 전체 step을 집계에서 제외

- 같은 request-group scope·실제 batch·P1/P2/기타-hit/miss 수 구성이 같은 step끼리 비교한다.
- Cap/clip·부분 batch를 먼저 제외하고, 표본5개 이상인 그룹의 중앙값 **2배 초과**만 한 번 제외한다.
  정확히2배는 유지하며, 구간별 cutoff나 반복적으로 줄이는 trimming은 하지 않는다.
- 시간·실제 반환 토큰·AL/hit event를 함께 제외한다. 출력·raw decode step 기록은 보존한다.
  Profile ON에는 이 duration filter를 적용하지 않는다.
- 콘솔 `Excluded long decode steps`, JSON `excluded_long_steps`/`duration_filter`,
  W&B `metrics_excluded_long_steps`로 제외 개수를 보고한다.
  A/B에서는 양쪽 retained step 교집합도 별도로 비교했다.
- Native 요약, benchmark 후처리, phase AL/hit가 같은 유효 모집단을 사용하도록 정리했다.
  과거 raw 로그의 수치와 filtered 요약을 구분한다.
- 파일: `ssd/ssd/utils/performance.py`, `ssd/ssd/engine/llm_engine.py`,
  `ssd/bench/bench.py`, `mlsys_measurement.py`, `mlsys_tree_results.py`.
- 상세 정책·회귀: [step_duration_filter_20261016/REPORT.md](results/mlsys_ver12/step_duration_filter_20261016/REPORT.md).

### E. 진단 benchmark·timeline 후처리

- `mlsys_phase_probe.py`: benchmark에서만 draft model+LM-head replay별 GPU marker를 넣고
  round 폭·후보별 node 수를 기록한다. Sampling/선택을 포함한 graph/K와 구별한다.
- `mlsys_phase_compilation.py`: 실제 compile/capture 구간을 따로 기록한다.
- `mlsys_local_phase_latency.py`, `mlsys_chain_phase_latency.py`: 로컬 진단 진입점이다.
  **orion 전용 guard가 있는 과거 캠페인 도구**이므로 다른 서버에서 그대로 실행할 일반 launcher가 아니다.
  원격 전용 campaign 스크립트도 역사적 재현용이며 내부 절대 경로를 그대로 실행하지 않는다.
- `ssd/tools/duet_timeline/`의 exporter·phase 평균·renderer·builder:
  optional forward probe/compile 구간/초기 제외 step 수를 지원하고,
  native chain은 각 forward의 실제 경계와 번호를 표시한다.
  과거 tree 전체 graph의 흰 분할선은 횟수 표시이며 개별 forward 실측 경계가 아니다.
- P1 hit/P2 hit/miss/전체별 step, rate, AL, Profile TPS, DS/PS 여유,
  forward1회 및 계산창(`forward 합 + signed 여유`)을 지원한다.
  여유를 전부 추가 forward에 쓸 수 있다는 보장은 아니다.
- 다음 target 요청 시작을 draft deadline으로 표시한다. 초록 원은 cache-ready,
  삼각형은 후처리 후 receive-ready다. Response wait 전체를 이전 PS에 귀속하지 않는다.
- Probe가 없는 자료는 graph/K로 남긴다. B>1 mixed-source를 임의로 단일 source로 표시하지 않는다.
  Generic 새 profile exporter는 현재 tree용이고 native chain 분석은 해당 campaign adapter를 쓴다.

함께 보존한 기존 진단:

| 폴더 (`results/mlsys_ver12/` 아래) | 의미 |
|---|---|
| `local_phase_latency_20261015` | 로컬7B/13B+68m, K3 tree의 W9/18/27·시간 예산·node 통계 |
| `chain40_k10_20261016` | 로컬 native chain40 roots/K10, source별 AL survival 및 두-step timeline |
| `tolist_review_20261016` | 이전 단순 tolist 생략 후보는 이득 없어 **미채택**, source archive도 감사 자료 |
| `response_wait_review_20261017` | 최적화 전 공백을 분해한 코드·trace 진단. 개선량의 증거가 아님 |
| `step_duration_filter_20261016` | 사용자 승인2배 집계 정책과 CPU 회귀 |
| `target_gap_optimization_20261017` | 이번 A/B/C 채택 근거·완료36실행·원시 압축파일 |

과거 PDF·JSON의 측정 버전·해시·GPU 표기를1.3으로 바꾸지 않았다.
새 코드가 들어갔다고 기존 그림이1.3 실측이 되는 것은 아니다.
미채택 tolist 후보·packed target·mixed-miss 정책·다른 서버에서 진행 중인 알고리즘은 새로 채택하지 않았다.

## 5. 검사와 재현

신규 핵심 검사는 `test_response_snapshot`, `test_single_verify_inputs`,
`test_deferred_tree_restore`다. 기존 `test_b_gt1_m2` fixture에 빠진
`tree_cache_is_tree` flag도 보완했다. Baseline에서도 실패하던 fixture였으며 serving 우회 수정이 아니다.
동일 fixture의 cache routing13개는 before/after 양쪽 통과했다.

- 최종 결합 runtime의 기존 원격 GPU 검사52개: raw archive의 `raw/checks_combined/`.
- 이번 전달의 CPU 회귀87개(12 suites), failure/error/skip0 및 소스 동일성: [checks/cpu/CHECKS.json](results/mlsys_ver13/checks/cpu/CHECKS.json).
- CPU 검사는 문서 전달 커밋 전 실행되어 부모 커밋 `b272892`를 기록한다. 실제 검사 소스 hash는
  같은 JSON에 있고, 전달 전 audit에서 최종 runtime/bench/tests/timeline Python 파일과 일치함을 확인한다.
- 보호한 원래 worktree와 runtime·전달 파일 검사: [RELEASE_AUDIT.json](results/mlsys_ver13/RELEASE_AUDIT.json).
- 이 숫자들은 겹치는 suite가 있으므로 더해서 독립 검사 수로 쓰지 않는다.
  과거 ver1.2의382개를 이번1.3의 새 GPU 검사 수로 쓰지 않는다.

저장소 루트에서 실행한다. Python3.11/3.12와 프로젝트 의존성, 그림 검사에는
numpy/matplotlib/pypdf가 필요하다. CPU 검사는 GPU 작업을 실행하지 않는다.

```bash
export DUET_PY=/path/to/ssd/venv/bin/python
"$DUET_PY" -B results/mlsys_ver13/validate_release.py --output /tmp/mlsys-ver13-cpu

# 해당 서버에 CUDA/FlashInfer 환경을 준비한 뒤 실행한다.
export SSD_HF_CACHE=/path/to/models
export SSD_DATASET_DIR=/path/to/datasets
export SSD_CUDA_ARCH=8.6  # 실제 GPU에 맞춰 변경
export CUDA_VISIBLE_DEVICES=0  # 다른 작업이 없는 GPU
"$DUET_PY" -B results/mlsys_ver13/validate_release.py --cuda --output /tmp/mlsys-ver13-cuda
```

CUDA 명령은 기존52개 suite와 forward-probe2개를 실행한다. 별도 process로 suite를 분리해
cache fixture의 import-time test flag가 다른 검사에 새지 않게 한다. 실패·오류·skip이 있으면
완료로 처리하지 않는다. `CHECKS.json`의 실제 건수와 로그를 확인한다.
Runtime을 병합해 바꾸면 identity 검사가 실패하는 것이 정상이다. 옛 manifest를 덮어써 통과시키지 말고
새 provenance·정확성 검사를 만들고 필요하면 성능 A/B를 다시 수행한다.
옛 `results/mlsys_ver12/validate_release.py`는1.2 고정 hash 검사이므로1.3 릴리스 검사에 쓰지 않는다.

원시 결과를 다른 서버에서 확인하려면:

```bash
cd results/mlsys_ver12/target_gap_optimization_20261017
sha256sum RAW_RESULTS.tar.xz
# ea03d76d7b622a02a77d80ea1b4ee1cebd9978757084855e5d64a25a66e6aebe
# raw/가 없는 새 checkout에서 실행. 원본 측정 서버는 필요 없다.
tar -xJf RAW_RESULTS.tar.xz
python analyze_steps.py --raw raw --output /tmp/ver13-step-comparisons.json
python analyze_profiles.py --raw raw --output /tmp/ver13-profile-components.json
```

압축파일에는659개 파일과36개 완료 run이 있다. `ARTIFACTS.json`, `ARCHIVE_CHECK.json`을 함께 본다.
기존 결과 JSON을 덮어쓰지 않고 재분석 결과는 다른 경로로 저장한다.

Timeline 도구는 [사용법](ssd/tools/duet_timeline/README.md)을 따른다. 새 trace에는 실제 commit과
하드웨어를 적고 기존 portable bundle은 원본 GPU 없이 다시 그릴 수 있다.

## 6. 다른 서버의 수신·검토·병합 순서

```bash
git fetch origin MLSys-ver1.3
git worktree add -b review/mlsys-ver1.3 ../PSD-MLSys-ver1.3-review origin/MLSys-ver1.3
cd ../PSD-MLSys-ver1.3-review
git rev-parse HEAD
git diff 899e4eb1..HEAD -- ssd/ssd
git diff 1a2e349..HEAD -- ssd/ssd/engine
git log --oneline 899e4eb1..HEAD
```

1. 이 문서에서A/B/C와 기존 승인D/E를 구별한다. `CHANGED_FILES.json`으로 누락을 확인한다.
2. `RUNTIME_MANIFEST.json`과 현재 runtime의 동일성, CPU 검사 결과를 확인한다.
3. 성능 근거는 개별 변경의 반복 A/B와 결합 출력 parity를 따로 읽는다.
   결합 개선율·새 최적 파라미터·full-dataset AL 개선을 입증한 릴리스가 아니다.
4. 다른 서버의 root/위치/tree 알고리즘을 병합할 때 `draft_runner.py`, `model_runner.py`,
   `verifier.py`, `speculator_async.py` 전체를 덮어쓰지 말고 변경 구간을 대조한다.
   특히 이전 KV state 수명, JIT 전 복원, 다음 forward 전 복원, response valid 길이를 지킨다.
5. 실제 하드웨어에서 CUDA 회귀와 대표 full-model parity를 실행하고 새 기록을 남긴다.
   새 알고리즘이 graph shape·root 수·M을 바꾸면 warmup과 최대 입력 capacity도 다시 확인한다.
6. 이후 필요 시 결합 반복 A/B, 70B·B>1·T=0 full-model 검증을 진행한다.
   이 항목은 미완료 적용 코드가 아니라 **새 조건에서 남은 검증 범위**다.

기존1.2 브랜치를 force-push하거나 원래 작업 폴더의 미커밋 파일을 정리하지 않았다.
이 브랜치를 검토한 뒤 수신 서버의 진행 중인 연구와 명시적으로 병합한다.
