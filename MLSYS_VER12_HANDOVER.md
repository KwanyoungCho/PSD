# MLSys ver1.2 — 채택한 PS 준비 최적화와 상세 timeline

> 이 문서는 ver1.2 당시 기록과 승인된 후속 진단을 보존한다. **현재 작업 기준은 `MLSys-ver1.3`**이며 [새 인수인계](MLSYS_VER13_HANDOVER.md)를 먼저 읽는다. 아래의 “미반영”, 작업 경로·성능 수치는 각 기록 당시 상태다.

2026-10-15 KST. **이 서버의 다음 작업 기준은 `MLSys-ver1.2`다.** 작업 폴더 이름은
`/home/chokwans99/PSD-MLSys-ver0.1` 그대로지만 실제 Git 브랜치는1.2다.
다른 서버에서 진행 중인 알고리즘은 이 기준 위에서 별도로 병합한다.

## 이후 성능 실험과 파라미터 검토 기준 — 2026-10-15 사용자 결정

- **새 성능 실험은 로컬 서버 `orion`의 RTX4090에서만 실행한다.** eslab18 등 다른 서버의
  과거 기록은 참고 자료이며 로컬 대조군을 대체하지 않는다. 각 비교는 같은 GPU 배치·모델
  dtype·입력 UID/길이·출력 상한·온도·seed·batch·runtime을 고정하고 변경 항목을 명시한다.
- 사용자가 파라미터를 하나씩 검토한 뒤 실험한다. 이 문서화 시점에는 새 실험을 시작하거나
  현재 설정을 변경하지 않았다. 기존 모델별 timeline의 설정을 최적값으로 간주하지 않는다.
- 새 tree 기준 실행은 DS/PS tree를 모두 명시적으로 켠다. 일반 Config의 P1 기본값은 off이므로
  tree preset과 혼동하지 않는다. B=1에서 먼저 검토하고 B>1은 별도 조건으로 평가한다.

| 검토 묶음 | 값과 정확한 의미 | Config 이름 |
|---|---|---|
| Exit 위치 | L: proxy를 얻는 target layer 인덱스(0부터) | `duet_exit_layer` |
| 순차 forward 횟수 | K1/K2: DS/PS의 forward round 수 | `duet_phase1_k`, `duet_phase2_k` |
| Cache root 수 | U1: DS 위치당 root 수 / R2: DS 중복 제거 후 PS 전체 root 수 | `duet_p1_roots_per_position`, `duet_p2_root_count` |
| Forward 입력 폭 | W1: DS 첫 회 이후 입력 수 상한 / W2: PS 매 forward 입력 수 | `duet_p1_forward_width`, `duet_p2_forward_width` |
| Root별 tree 노드 상한 | M1/M2: root 토큰을 제외한 DS/PS tree의 생성·검증 노드 상한 | `duet_p1_tree_verify_nodes`, `duet_p2_tree_verify_nodes` |
| 자식 수 | C: 부모 하나의 분포에서 샘플링하는 자식 수 상한 | `duet_tree_c_tensor` |

DS 첫 forward는 모든 root R1을 평가하며 이후 입력 폭은 min(R1,W1)이다.
R1은 현재 처리하는 위치 수×U1이므로 실제 `round_widths`를 기록한다. PS는 R2≤W2이다.
Root는 서로 다른 cache 결과를 대비한 시작 후보, W는 한 번에 계산하는 행 수, M은 후보
하나가 hit됐을 때 검증하는 후속 노드 수다. 이 셋을 같은 예산으로 표시하지 않는다.
M은 G=M 계약에 따라 생성 도중에도 적용되며, 생성 후 임의 rerank를 추가한 것이 아니다.

처음 비교는 현재 기본인 `last` 확장, calibration 없는 draft 경로 점수, root `e(1-q)`,
`topm` 정규화, 위치 혼합 비중0.25를 고정한다. `last/frontier` 비교는 이후 별도 알고리즘
비교로 분리한다. 과거 timeline에는 frontier 실행도 있어 설정을 그대로 섞지 않는다.
`duet_tree_beta`는 현 dynamic 실행기의 예산 배분 분기에서 사용하지 않으므로 이 탐색 대상이 아니다.
G, 위치별 top-L, 전송 후보 여유분은 자동 파생하고 threshold는0을 유지한다.

검토 순서는 입력 폭·root 수·M·C의 출발점을 고정한 상태에서 L/K1/K2의 시간 배치를 먼저
확인하고, root 수 → 입력 폭 → M → C를 검토한 뒤 변경된 비용으로 L/K1/K2를 재확인한다.
L/K1/K2는 서로 연결되므로 한 값만 바꾼 결과와 조합의 결과를 구분한다. 각 단계에서
DS/PS 준비·graph·정리, deadline 여유, target 검증 시간을 breakdown으로 확인하고,
P1/P2/miss 비율·조건부 AL·전체 AL·TPS도 같은 모집단으로 함께 보고한다. Profile OFF의
성능 평가와 profile ON의 구간 진단은 구분한다. 입력별 문맥 길이와 실제 query 수도 기록한다.

## 이번 전달에서 바뀐 것

| 분류 | 반영 내용 | 근거/주의점 |
|---|---|---|
| 실행 코드 | PS root 매핑 재사용, 조상 마스크 GPU 유지, 여섯 유효성 검사를 한 kernel로 통합 | PS 전체 5.269→4.928ms, -6.48%; Llama2-7B+AMD135M, B1/4090의 고정 설정 |
| 실행 코드로 미채택 | 추가 fan-out readback 지연·host 문맥 길이 재사용 | 추가 차이 -0.017ms의 CI가0 포함. 후보와386개 검사 기록은 연구 자료로만 보존 |
| 분석 도구 | 기존 상세 이벤트/색/무늬, PS 준비 A/B, 미계측 간격, 명령 수신 | 새로운 runtime 계측을 추가한 것이 아닌 기존 profile 후처리 |
| 두-step 그림 | 같은 요청의 연속 두 step, 다음 target 요청 시작 deadline, cache/receive-ready 여유 | 원시 시간 보존. 다른 서버에서 JSON만으로 재생 가능 |
| 평균 표·새 profile 이식 | P1 hit/P2 hit/miss의 발생 비율·AL, 전체 hit·AL·Profile TPS, step·DS/PS 시간·여유·graph/K, 새 profile exporter | 동일한 유효 step 모집단, 표본 수 공개. 원본 서버 없이 재생성 검사 |

이번1.2는 **후보 수식, 위치 수식, tree 확장 정책, K1/K2, 입력 폭, G=M 계약을 바꾸지 않는다.**
기존1.1에 들어 있던 알고리즘을 유지한다. 새 사용자 옵션도 없다. 파라미터는
[ver1.1 문서](MLSYS_VER11_HANDOVER.md)를 따르며 이번 latency 비교 설정을 모델별 최적값으로 사용하지 않는다.

## 브랜치 받기와 검토 순서

저장소는 `https://github.com/KwanyoungCho/PSD.git`, 받을 브랜치는 **`MLSys-ver1.2`**다.
기준은 `MLSys-ver1.1@0916871`이다. 기존 서버 작업을 덮어쓰지 않도록 별도 worktree에서 검토한다.

```bash
git fetch origin MLSys-ver1.2
git worktree add -b review/mlsys-ver1.2 ../PSD-MLSys-ver1.2-review origin/MLSys-ver1.2
cd ../PSD-MLSys-ver1.2-review
git rev-parse HEAD
git log --oneline 0916871..HEAD
git diff 0916871..HEAD -- ssd/ssd
```

커밋은 실행 코드, 도구/검사, 보고서/실험 증거 순서로 나눴다. 실행 코드만 검토할 때는
**`9390e21`**을 본다. 바뀐 runtime 파일은 `draft_runner.py`, 새 `helpers/p2_prepare.py` 두 개다.
`test_p2_preparation.py`의12개 검사가 같은 커밋에 있다. runtime91파일 전체의 해시가
기존382개 회귀 통과 당시 소스와 일치한다. 외부 서버의 `draft_runner.py` 전체를 덮어쓰지 말고
이 변경 구간만 비교하여 충돌을 해결한다.

검토할 자료:

1. [최종 반영 보고서](results/mlsys_ver12/REPORT.md): 변경, 측정 범위, 남은 비용, 그림 해석.
2. [채택 근거](results/mlsys_ver11/ps_preparation_20261015/REPORT.md): 동일 step breakdown, 세 반복, profile OFF parity.
3. [추가 후보 미채택 근거](results/mlsys_ver11/ps_device_20261015/REPORT.md): CI·후보 소스·복원 증거.
4. [버전1.2 검사](results/mlsys_ver12/checks/CHECKS.json): runtime 동일성,31개 offline 검사,최초 전달의16쪽 두-step 그림 검증; 현재17쪽 모델별 모음은 `model_timelines/checks/CHECKS.json` 참조.
5. [timeline 사용법](ssd/tools/duet_timeline/README.md): 새 실험 exporter, 표준 outcome JSON, 저장된 그림 재생 명령.

과거 실험 보고서의 “작업 트리”, “새 커밋·push 없음”은 **측정·보고 당시 기록**이므로
원본/해시 manifest를 보존했다. 현재 채택 및 전달 상태는 이 문서를 기준으로 확인한다.
`ps_device_20261015/candidate_source` 아래 코드는 미채택 후보 snapshot이며 production import 경로가 아니다.
그 파일이나 `RUNTIME_INCREMENTAL.patch`를 현행 runtime에 덮어쓰지 않는다.

## 검증 재현

이미 GPU 회귀 **40 suites/382개, 오류·실패·skip0**을 통과한 runtime과 동일하다.
최초 전달의 timeline 검사는31개다. 평균 표·서버 이동 기능을 추가한 이번 전달에서는
동일성 해시 확인 및 timeline 관련 **42개 CPU 검사가 모두 통과했다**(오류·실패·skip 0).
실제 797 step의 새 exporter 재집계도 기존 네 행의 평균·rate·AL·TPS와 일치했다.
최종 기록은 `model_timelines/checks/CHECKS.json`과 `EXPORT_REPLAY.json`을 확인한다.
GPU와 CPU 검사에는 중복 suite가 있으므로 합산해서 독립 검사 수로 쓰지 않는다.
미채택 후보의386개 검사는 배포 runtime의 검사 수가 아니다.

```bash
export DUET_PY=/path/to/venv/bin/python
# GPU 불필요: numpy, matplotlib, pypdf가 있는 Python
"$DUET_PY" -B results/mlsys_ver12/validate_release.py --output /tmp/mlsys-ver12-offline

# 새 하드웨어로 이식한 후에는 해당 서버의 CUDA/FlashInfer 환경에서 실행
export PYTHONPATH="$PWD/ssd:$PWD/ssd/tests"
export PYTHONDONTWRITEBYTECODE=1
# CUDA_VISIBLE_DEVICES, SSD_CUDA_ARCH, SSD_HF_CACHE, SSD_DATASET_DIR는 서버에 맞게 설정
"$DUET_PY" -B results/mlsys_ver11/check_suites.py --output-dir /tmp/mlsys-ver12-cuda
```

`validate_release.py`는 이 릴리스의 runtime 동일성을 검사한다. 다른 알고리즘을 병합한
뒤에는 이 검사가 실패하는 것이 정상이며, 이후 소스는 새 provenance와 회귀 결과로 검증한다.
원래 checks/raw를 새 실행 결과로 덮어쓰지 않는다. 예전 검사 건수만 기대하는 대신
실제 suite/test/skip/실패·오류 수와 종료 코드, `source_unchanged`를 확인한다.

## 그림부터 확인하기

**[모델별 timeline 보고서](results/mlsys_ver12/model_timelines/REPORT.md)를 시작점으로 사용한다.**
[전체 PDF](results/mlsys_ver12/model_timelines/ALL_MODELS.pdf)와 모델 조합별 PDF5개로 통일했다.
CodeLlama의 두 설정은 같은 PDF에 묶었다. 이전 모델별 timeline/별도 예시의 중복 파일은 삭제했다.
원시 JSON·CSV와 실험별 A/B 증거는 보존했다. 새 그림은 기존 profile의 재시각화다.
최초 전달의 `EVIDENCE_MANIFEST.json`과 `checks/`는 Git `1f87514` 당시의 기록이다.
그림 정리 이후에는 `model_timelines/INDEX.json`, `CLEANUP.json`, `checks/CHECKS.json`을
사용한다. 삭제한 옛 그림이 최초 manifest에 남아 있다는 이유로 raw 손실로 판단하지 않는다.

빨간 점선은 **다음 target 요청 시작**이다. 초록 원은 PS cache 완성, 삼각형은
후처리를 끝내고 draft가 다음 명령을 받을 준비를 마친 시각이다. 두 시각을 구분한다.
회색 command 수신 구간 전체가 추가 forward를 넣어도 되는 여유 시간은 아니다.
표의 ready margin은 관측된 차이이며, cross-GPU 시각 정렬 오차 및 시스템 변동이 있어
그만큼 계산을 추가해도 target stall이 없다는 보장이 아니다. 다음 response wait 전체도
이전 PS에 귀속하지 않는다.

그림의 표는 이제 **P1 hit/P2 hit/miss/전체 평균**이다. 대표 두 step만의 평균이 아니며
전체는 실제 step 수로 가중한다. Step 길이, DS/PS 전체 latency, DS/PS 여유, graph/K1·K2와
표본 수를 표시한다. DS 여유는 proxy 전송 준비 완료 기준, PS 여유는 다음 target 요청
시작−draft 수신 준비 기준이다. Graph/K는 sampling·선택을 포함하고 개별 forward 실측은 아니다.
Rate는 전체 step 중 P1/P2/Miss의 비율이며 Overall 행은 전체 hit rate다. AL은 recovery/bonus
포함, Profile TPS는 같은 유효 step의 실제 반환 토큰 합÷step 시간 합이다. 전체 Spec-Bench나
profile OFF TPS로 인용하지 않는다. 토큰 기록이 없는 새 입력은 해당 지표를 N/A로 표시한다.

Llama2-7B의 두 draft 조합은 **draft만 바꾼 대조 실험이 아니다**. llama-68M은 RTX3090·과거 v1.0·
4턴·출력 상한256, AMD135M은 RTX4090·채택 PS 최적화·24턴×3회·출력 상한96의 기록이다.
유효 step 평균은 각각25.411/19.905ms, 논리적 query 수는9.657/9.620개, 기존 KV 문맥은
466.676/203.325 tokens다. Query 수가 크게 늘어난 것은 아니며, 문맥·GPU·코드·입력 차이가
함께 있다. `PHASE_AVERAGES.json`의 `workload`에 경우별 query 분포·문맥·target 구간 평균을
보존했다. 이 자료만으로 latency 차이를 draft 모델이나 특정 최적화 하나의 효과로 귀속하지 않는다.

동봉된 기존 trace 3,700 step에서는 proxy 계산과 target 후반 forward의 중첩이 없었다.
새 profile은 중첩 건수와 시간을 다시 계산한다. `exit_proxy_side`라는 이름만으로 별도
stream을 사용했다고 판단하지 않는다. 전송 enqueue는 통신 완료 계측도 아니다.

## 다른 서버에서 새 timeline 만들기

저장된 그림은 다음 명령으로 재생성한다. GPU·모델·원본 서버 profile이 필요 없다.

```bash
python ssd/tools/duet_timeline/build_model_timelines.py \
  --manifest results/mlsys_ver12/model_timelines/SOURCES.json \
  --output /tmp/duet-model-timelines
```

새 실험의 profile과 실제 cache source 기록을 쓰려면 README의 `timeline_inputs.json`을
작성한 후 다음 명령 하나를 실행한다.

```bash
python ssd/tools/duet_timeline/export_profile_timelines.py \
  --spec /path/to/timeline_inputs.json \
  --output /path/to/new_timeline_bundle
```

현재 exporter는 **B=1, DS/PS tree trace**용이다. B>1의 혼합 source를 임의로 하나로
분류하지 않으며, 그 경우는 별도 집계 계약이 필요하다. 기존 benchmark `run.json` 또는
README에 명시한 표준 outcome JSON을 입력한다. 새 모델·반복 수·서버 경로는 코드 수정 없이
spec으로 지정한다. 측정 commit·환경·설정 설명을 실제 실험에 맞게 남긴다.
출력 폴더 전체가 이식 가능한 묶음이다. 원본 서버 경로는 출처로만 남고 재생성 시 접근하지 않는다.
이번 추가 변경은 분석 도구·그림·문서이며 inference runtime과 실행 옵션을 바꾸지 않는다.

## 다른 서버에서 후속 확인할 것

- 70B/Blackwell 및 B>1의 **PS 준비·PS 전체 latency**를 같은 입력/출력/phase 조건으로 재측정한다.
  이번6.48%를 그 환경의 수치로 사용하지 않는다.
- 변경된 tree 알고리즘과 병합하면 root 매핑·조상 마스크 shape, 실제 root R와 forward lane W,
  fallback 경로의 계약을 다시 검사한다. 실패 우선순위를 보존하는 유효성 검사를 생략하지 않는다.
- PS 전체는 proxy 수신 완료→merge 완료다. 현재 최적화는 **draft 측 준비 overhead** 개선이며,
  Transformer 자체/target forward 가속, TPS/AL 증가, 최적 파라미터 확정 결과가 아니다.
- 새 graph key 준비는 기존 prewarm 정책을 유지한다. 후속 비교에서도 준비 영향 step을 공개한다.

## 로컬 DS/PS 시간 예산과 입력 폭 진단 (2026-10-15)

사용자가 직접 파라미터를 검토하기 위한 첫 측정은
[local_phase_latency_20261015/REPORT.md](results/mlsys_ver12/local_phase_latency_20261015/REPORT.md)에 정리한다.
완료 판정과 검증은 같은 폴더의 `CHECKS.json`, 입력/계획은 `INPUTS.json`/`PLAN.json`에 있다.
기존 여러 서버의 최적 설정 timeline과 혼합하지 않는다.

- 로컬 orion RTX4090에서 LayerSkip-Llama2-7B/13B + JackFram/llama-68m만 측정한다.
  7B는 target1장+draft1장,13B는 target2장+draft1장이다.
- B1, T0.7, BF16, exit21/26(0-based), K1=K2=3, C3, G=M40.
  DS 위치당 root3, PS 전체 root9. W1/W2는9/9에서 한쪽만18/27로 바꾼다.
- 7B DS는 GPU4·5, PS는6·7에서 각자9/9 기준과 비교한다.13B는4·5·6이다.
  원격 서버 성능 데이터를 섞지 않는다.12개 사전 선택 입력×2회이며 full Spec-Bench 품질 평가가 아니다.
- Benchmark 전용 GPU marker로 **model+LM head의 실제 forward**를 측정했다.
  graph/K에는 sampling/선택도 포함되므로 다른 지표다. Serving runtime은 변경하지 않는다.
- DS 첫 forward는 모든 위치 root를 처리하므로 W9가 전체9행이라는 뜻은 아니다.
  M40에서 최대123행, 짧은 bucket은12행이고, 후속 forward는min(bucket roots,W)다.
  C3/K3의 root 최대 자손은39개라 M40은 생성 제한에 걸리지 않지만 padding 비용은 발생한다.
- 첫10step/종료·cap step/실제 컴파일·capture 영향을 제외한다.13B 최초 rank1 커널 컴파일 대기를
  정상 반복 시간에 포함하지 않는다. 빠른 step만 골라내는 latency cutoff는 없다.
- 7B base의 telemetry 수집 공백을 공개하고 같은 설정을 추가 반복한다.
  기존 baseline을 더 빠른 반복으로 교체하지 않으며 출력·시간 차이는 `AUDIT.json`에 보존한다.

새 portable timeline은 `local_phase_latency_20261015/timelines/ALL_MODELS.pdf`와 `index.html`이다.
두 모델×P1 hit/P2 hit/miss의6개 연속 두-step 그림에 같은 전체 표본의 평균을 붙인다.
기존 exporter도 optional forward probe와 초기 제외 step 수, 컴파일 구간 제외를 지원한다.
Probe가 없는 과거 자료는 계속 graph/K로 표시하므로 실제 forward 실측으로 오인하지 않는다.
그림의 흰 점선은 횟수 표시이며 개별 forward의 실제 경계선은 아니다.

이 진단에서 계산창은 `실제 forward 합 + 관측된 signed 여유`다.
현재 K3의 나머지 작업을 고정한 값이므로, 이를 forward1회로 나눈 수를 새 K의 보장값으로
사용하면 안 된다. K를 늘릴 때 sampling/선택·tree 크기·M 제한도 다시 검토한다.

## Native chain 40후보 / K1=K2=10 진단 (2026-10-16)

[chain40_k10_20261016/REPORT.md](results/mlsys_ver12/chain40_k10_20261016/REPORT.md)에
동일한 두 모델의 새 측정과 AL survival curve를 보존했다. 완료 근거는 같은 폴더의
`CHECKS.json`, 조건은 `PLAN.json`/`README.md`, 실제 전달한 설정은 `RUN_PROVENANCE.json`이다.

- 기존 native chain을 실행하며 tree executor의 C1 대용 경로를 사용하지 않았다.
  DS 총40개는11개 위치에 `[4,4,4,4,4,4,4,3,3,3,3]`, PS는 기존 점수와 DS 중복 제거 후40개다.
  각 phase의 입력40행×10회와 target 검증 길이10을 모든 step의 CPU shape/event로 확인했다.
- 로컬7B GPU4·5,13B GPU4·5·6에서 순차 실행했다. B1/BF16/T0.7/exit21·26,
  직전과 같은12입력×2회, 입력512/출력192 상한이다. Full Spec-Bench AL 검증은 아니다.
- Warmup/초기10step/종료·cap·clip/컴파일 영향 제외 후7B1499,13B1530step이다.
  P1/P2/miss 평균 AL은7B2.669/2.240/1.739,13B2.613/2.191/1.798이다.
  `P(AL>=x | source)`를1~11에 대해 그렸다. AL은 correction/bonus를 포함하며
  각 source의 조건부 확률이다. 초기 구간을 포함한 민감도와 UID bootstrap 구간도 CSV에 있다.
- `timelines/ALL_MODELS.pdf`는2모델×3source의6페이지다. 각 페이지는 연속 두 step이며,
  DS/PS10회 각각의 native replay를 실측 경계와 번호로 표시한다. Timeline 도구는 과거
  tree 전체 graph의 균등 횟수 표시와 native chain의 실측 구간 표시를 구분한다.
- 두 모델 모두 DS/PS forward1회 약0.214ms. PS 여유 평균은7B−2.130ms,13B−1.099ms다.
  K10은 요청한 진단 조건이며 최적 설정으로 채택한 값이 아니다. 기존 tree 진단과는
  root/K/M까지 달라 구조만의 차이에 대한 A/B 비교로 인용하지 않는다.
- Serving runtime 변경 없음. Reporting 검사26개 통과, 할당 GPU에서 외부 process 관측 없음.
  원시는 `/data/chokwans99/duet-chain40-k10-20261016/`에 남겼다.

## `tolist()` 생략 시험 — 무효 / 미채택 (2026-10-16)

[tolist_review_20261016/REPORT.md](results/mlsys_ver12/tolist_review_20261016/REPORT.md)에
동일 GPU의 before/after/after/before 시험을 보존했다. 원래 runtime은 변경하지 않는다.
13B+68m, B1/K3/C3/W9/G=M40에서 출력48/48 동일, 검사25개 통과였지만,
PS 준비3.153→3.193ms, PS 전체4.799→4.843ms로 개선이 확인되지 않았다.
사용자가 무효 처리를 요청하여 **candidate는 미채택**이다. 후속 실험은 원래
`MLSys-ver1.2`를 계속 사용한다. 미채택 patch와 frozen source는 감사 자료이며
production 반영 목록에 넣지 않는다. 시험용 detached worktree는 정리한다.

## 향후 성능 집계: 전체 step 중앙값의 2배 초과 제외 (2026-10-16)

사용자가 이후 논문 성능 측정에도 적용하도록 지정했다. 상세 규칙과 검사는
[step_duration_filter_20261016/REPORT.md](results/mlsys_ver12/step_duration_filter_20261016/REPORT.md)에 있다.

- Profile OFF에서 같은 request group·실제 batch·P1/P2/기타 hit/miss 구성끼리 비교한다.
  표본5개 이상이면 중앙값의 **2배 초과**인 전체 step만 한 번 제외한다. 구간별 검사는 하지 않는다.
- 그 step의 시간·토큰·AL/hit event를 함께 빼며, 원시 기록·출력은 유지한다.
  기존 cap/clip·부분 batch 제외는 유지한다. 모든 비교 방법에 같은 규칙을 적용한다.
- 콘솔 `Excluded long decode steps`, JSON `summary.excluded_long_steps`,
  W&B `metrics_excluded_long_steps`로 제외 개수를 보고한다.
- A/B 후처리는 옛 `PAPER_CHECKOUT`의 집계 구현과 무관하게 현행 공통 모듈을 사용한다.
  과거 보고서를 조용히 새 정책으로 바꾸지 않는다.
- 추론 알고리즘·runtime hot path 최적화가 아닌 후처리 변경이다.
  사용자의 지시대로 모델 실험은 나중에 진행하고, 이번에는 CPU 회귀 검사만 실행한다.

## Target forward 사이 전체 공백 검토 (2026-10-17)

[response_wait_review_20261017/REPORT.md](results/mlsys_ver12/response_wait_review_20261017/REPORT.md)에
기존 trace와 코드의 분석을 보존했다. 사용자가 지칭한 wait은 개별 recv marker가 아니라
이전 final head 완료→다음 target Transformer 시작의 전체 공백이다.

- P1-hit 기준 약7.3~9.2ms. 수락 판정 약1.5~1.8ms, draft 응답 약2.8~3.8ms,
  수신 후 다음 실행 준비 약1.5~2.9ms 등이 직렬로 들어간다.
- Tree K3에서는 PS 대기보다 요청 처리/KV 복구·tree 포장과 target 입력 재구성이 중요하다.
  Chain K10은 PS 초과도 실제 대기에 포함된다.
- 우선 검토는 중복 metadata readback 제거, hit 응답 후 draft KV 복구, 확정된 tree wire 사전 준비다.
  B1 tree 수락은 현재 Python 제어의 tensor 보행이며 GPU scalar read가 남아 있다.
- GPU 실험·runtime 변경·신규 채택 없음. 이번 수치는 개선량이 아닌 기존 profile의 병목 진단이다.
