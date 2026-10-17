# 로컬 GPU4–7 추가 최적화 검증

2026-10-17 KST. 완료20실행. **유지한 것은 중복 준비 제거만이다.** 기존 production 브랜치와 이전 실험 브랜치는 보존했고, 새 `exp/duet-local-followup-20261017`에 검토용 변경을 두었다. 작업 브랜치 commit/push는 하지 않았다. 실행용 독립 snapshot에만 source 동결용 로컬 커밋을 만들었고 VARIANTS.json에 기록했다.

## 결론과 response wait의 범위

모든 후보가 response wait에 속하지는 않는다. q 수신 겹치기는 그 span의 직접 대상이지만 전체 step 이득이 반복에서 확정되지 않았다. Graph 세분화는 response wait 뒤의 target graph_pre/post를 줄이려는 변경이고, 현재 M15에서는 이득이 매우 작고 별도 검증에서 재현되지 않았다. **새로 유지한 코드는 response wait 뒤의 CPU 검증 준비와 early-exit proxy 입력 준비를 줄인다.**

같은 엔진에서 무작위로 경로를 교대한 검사에서는 현재 step 효과가 **-0.038ms**, 95% CI [-0.056, -0.023]ms였다. 약19ms step 중 보수적으로 **약0.04ms, 0.2% 수준의 작은 개선**으로 설명한다. 프로세스별 A/B에서 관측된0.49–0.84% 전부를 코드 자체 효과로 귀속하지 않는다.

## 실제 유지한 코드

1. `TreeProxyCUDAGraph.replay`: exact-width 입력은 모든 원소를 copy로 덮어쓴다. 그 직전에 수행하던 exit/q/token buffer 초기화3개를 생략했다. 부분-width 호출에는 기존 zero tail 처리를 유지한다.
2. `Verifier`→`ModelRunner`: rank0에서 검증한 동일한 **불변 tuple packet**의 파싱 결과를 같은 호출 동안 재사용한다. 최초 wire 검증은 유지하며 TP follower·직접 호출·다른 packet·mutable list는 원래 검증을 거친다. 캐시는 finally에서 지운다.
3. Captured proxy graph 경로에서 사용하지 않던 부모·형제 Python 목록 생성을 eager fallback 안으로 이동했다.

Root 점수 e(1-q), 위치 혼합0.25, tree 선택 q-path/last, sampling·수락 법칙, 온도, 전송 형식, KV 의미와 반환 토큰은 바꾸지 않았다. 추가 사용자 옵션이나 추가 graph bucket도 넣지 않았다. [선택한 코드 patch](PREP_RUNTIME.patch).

## 고정 실험 조건

로컬 orion RTX4090 GPU4/5와6/7의 PIX 쌍을 사용했다. Target1GPU+draft1GPU, LayerSkip-Llama2-7B + JackFram/llama-68m full BF16, B1, T0.7. K1/K2=8/5, exit22(0-based), DS 위치당4roots·후속입력32, PS15roots·입력15, C3, G=M1=M2=15다.

이 파라미터는18번 서버에서 찾은 권장값을 고정해 코드 효과를 비교한 것이다. 로컬 서버의 최적 파라미터를 새로 찾았다는 뜻은 아니다. 과거 RTX3090·다른 K의 개선율과 더해서 제시하지 않는다.

기본 진단은 고정12개 SpecBench 입력, input cap512, output cap256, process마다2passes와 warm3이다. 12입력 반복을 독립24문항으로 세지 않았다. 별도 검증은 겹치지 않는48문항/56raw턴, input cap0, output256, seed42다. MT 두 턴은 bootstrap에서 같은 question cluster로 묶었다. Cold-prefix 조건을 모든 arm에 동일 적용했다. Full480/560 또는 output1024 평가가 아니다.

## Profile OFF 전체 step

| 후보 | 진단 전→후(ms) | 감소율, 95% CI | process 반복1/2 | 별도56턴 감소율, 95% CI | 판정 |
| --- | --- | --- | --- | --- | --- |
| 검증 graph 세분화 | 19.077 → 19.053 | 0.12% [0.01, 0.21] | 0.06% / 0.18% | -0.28% [-0.61, 0.07] | 미채택 |
| q 수신 겹치기 | 19.044 → 19.022 | 0.11% [-0.07, 0.26] | 0.34% / -0.11% | 미실행 | 미채택 |
| 중복 준비 제거 | 19.150 → 18.989 | 0.84% [0.71, 0.96] | 0.77% / 0.91% | 0.49% [0.33, 0.66] | 유지 |

진단은 같은 GPU쌍에서 process를 바꿔 A→B→B→A로 비교했다. Prep 첫 A는 직전 bucket 실험의 마지막 baseline 실행을 재사용했다. 같은 baseline을 별도 새 실행으로 세지 않았다. 서로 다른 process에서는 변경하지 않은 miss 경로도 함께 움직였으므로, 아래의 같은 엔진 검사가 원인 분리의 보강 자료다.

각 step의 출력·source/hit/valid_k/수락 길이가 동일한 비교만 사용했다. 모든 후보와 별도56턴 검증이 이 parity를 통과했다. Cap/clip 및 기존 동일 request/source/batch median의2배 초과 step을 양 arm에서 함께 제외한 교집합이다. 준비 시간을 retained step에서 임의로 빼지 않았다. 별도 CPU perf_counter도 native step 감소 방향을 확인했다.

### 유지한 코드의 경로별 진단

| 경로 | step 전→후(ms) | target 호출+수락 전→후(ms) | 표본 |
| --- | --- | --- | --- |
| P1 hit | 18.954 → 18.805 | 17.309 → 17.240 | 1842 |
| P2 hit | 18.961 → 18.806 | 17.313 → 17.238 | 1834 |
| Miss | 20.572 → 20.343 | 17.376 → 17.304 | 498 |
| 전체 | 19.150 → 18.989 | 17.319 → 17.247 | 4174 |

Miss는 새 GPU 준비 변경을 직접 받는 경로가 아니다. 이 표의 miss까지 빨라진 수치를 코드 효과로 주장하지 않는다. 해당 process 간 공통 변동 때문에 전체0.84%를 순수 코드 효과라고 단정하지 않는다.

## 같은 엔진 내 무작위 교대

동일 model·CUDA graph·allocator를 둔 하나의 엔진에서 기존/개선 Python 진입점을 무작위로 바꿨다. 교대는 측정 시작 전에 하고, 독립 RNG를 써 proposal/verification RNG를 건드리지 않았다. 12입력×4passes=48requests이며 앞24개의 출력·수락은 원본 baseline과 동일했다. 이 adapter는 분석용이며 유지한 runtime에 들어가지 않는다.

UID/pass/source/valid nodes/context128/직전source를 맞추고 현재 mode와 직전mode 효과를 함께 추정했다. 실제 채택용 runtime은 매 step 개선 경로만 사용한다.

| 분석 | 사용 step | 현재 step 효과(ms), 95% CI |
| --- | --- | --- |
| 전체 | 2038 | -0.0384 [-0.0561, -0.0229] |
| P1 hit | 941 | -0.0404 [-0.0589, -0.0211] |
| P2 hit | 1024 | -0.0440 [-0.0726, -0.0194] |
| Miss | 73 | 0.0687 [-0.0294, 0.1976] |

음수가 단축이다. Miss는 matched73step/7UID로 작고 현재 효과 CI가0을 포함하므로 개선·악화 어느 쪽도 확정하지 않는다. One-step lag를 포함한 전체 합계는 -0.0412ms이며, 이것도 별도의 pipeline 모델 가정이 있는 보강 추정이다. 전체 current CI의 질문 단위 cluster 수는12다. [원시 추정](MIXED_ANALYSIS.json), [그림](WITHIN_ENGINE.pdf).

## 직접 component 측정과 그래프

같은 GPU에서 고정 입력을 ABBA로 교대하여 proxy 입력+replay와 CPU metadata 처리를 따로 측정했다. Proxy serial 값은 매 호출 GPU 완료까지 기다린 경우, pipelined 값은 여러 호출을 enqueue한 뒤 기다린 경우다. 실제 모델 전체 개선율로 대체하거나 이 시간들을 단순 합산하지 않는다.

| 노드 | Proxy serial 전→후(ms) | CPU metadata 전→후(µs) |
| --- | --- | --- |
| 3 | 0.3481 → 0.3357 | 6.75 → 1.89 |
| 6 | 0.3543 → 0.3447 | 6.62 → 1.46 |
| 15 | 0.3855 → 0.3760 | 11.22 → 1.44 |

[전체 step 비교](WHOLE_STEP.pdf) · [구간별 비교](BREAKDOWN.pdf) · [전후 연속 step timeline](timelines/ALL_MODELS.pdf). Transformer graph_pre/post 자체가 빨라졌다고 주장하지 않는다. 유지한 변경의 해당 GPU 구간은 대체로 같고, 줄인 것은 그 밖의 준비 작업이다.

### Response wait 해석

q-overlap 후보의 response span은 q enqueue 시점에 끝나고, 실제 q 의존성은 proxy/accept 소비 시점에서 보장한다. 따라서 작은 response span만으로 통신이 빨라졌다거나 전체 step이 그만큼 줄었다고 할 수 없다. 지연 sender를0.4초 둔 별도2GPU 검사에서는 독립 계산이 먼저 완료되고 두 소비 stream에서 q가 맞게 읽히는 것을 확인했지만, 일반 inference에서 전체 이득은 확정되지 않아 미채택했다.

Graph 세분화는4/8/16행으로 실측 입력 padding을 줄였지만 작은7B batch에서는 forward 시간이 행 수에 비례하지 않았다. 예를 들어 valid3의 graph_pre+post는15.255→15.186ms로 약0.069ms 줄었고 valid6에서는 약0.005ms였다. 전체 평균에서 효과가 희석됐으며 별도56턴에서는 -0.28%로 방향도 유지되지 않았다. 추가 graph 메모리·준비 비용을 감수하며 default에 넣지 않았다.

## 기능·성능 기록 검증

완료 모델 실행20건(기본 진단·반복11, profile5, 별도56턴3, 무작위교대1)의 source/bench 상태·실제 반환 토큰·native/독립 wall 기록과 GPU telemetry를 검증했다. 각 GPU에 외부 compute process가 함께 잡힌 모델 실행은 없었다. 근거는 VALIDATION.json이다.

최종 선택 코드에서 **20개 CPU/CUDA 검사 통과, skip0**. 잘못된 packet·mutable list·다른 packet은 기존 검증을 거치는지, 동일한 불변 packet만 재사용하는지 확인했다. Proxy 입력에 garbage를 넣고 full/partial width를 연달아 실행해 zero-before-copy와 입력·출력이 bitwise 동일한지도 검사했다. 추가 q 지연 수신 검사는 미채택 overlap 후보에 대한 것이다.

새 테스트의 첫 시도는 inference tensor를 테스트에서 inference_mode 밖에서 수정한 fixture 오류였다. 수정 후 최종20개를 다시 통과했고 첫 로그도 PREP_TESTS_FIRST_FAILURE.log에 보존했다. 모델 runtime 실패를 숨긴 것이 아니다. Profile5개도 각각 baseline의 같은 입력 출력·수락과 모두 일치했다.

| 실행 | >2배 긴 step 제외 | cap/clip 제외 | 생성0턴 |
| --- | --- | --- | --- |
| baseline_pair0_a | 9 | 20 | 0 |
| baseline_pair0_d | 0 | 20 | 0 |
| baseline_pair0_prep_d | 0 | 20 | 0 |
| baseline_pair0_profile | 0 | 9 | 0 |
| baseline_pair1_a | 9 | 20 | 0 |
| baseline_pair1_d | 0 | 20 | 0 |
| baseline_pair1_profile | 0 | 9 | 0 |
| buckets_pair0_b | 0 | 20 | 0 |
| buckets_pair0_c | 0 | 20 | 0 |
| buckets_pair0_profile | 0 | 9 | 0 |
| heldout_baseline | 4 | 36 | 0 |
| heldout_buckets | 0 | 36 | 0 |
| heldout_prep | 0 | 36 | 0 |
| mixed_prep | 0 | 37 | 0 |
| overlap_pair1_b | 1 | 20 | 0 |
| overlap_pair1_c | 0 | 20 | 0 |
| overlap_pair1_profile | 0 | 9 | 0 |
| prep_pair0_b | 0 | 20 | 0 |
| prep_pair0_c | 0 | 20 | 0 |
| prep_pair0_profile | 0 | 9 | 0 |

Profile ON은 긴 step 필터를 적용하지 않는다. 초기화·graph capture는 decode 집계 밖이고 raw는 모두 보존했다. 특이 step을 제거한 횟수와 raw time을 숨기지 않는다. 다른 서버·GPU·모델·B>1·greedy full-model 성능으로 일반화하지 않는다.

## 코드 검토·후속 작업

기준은 기존 사전 포장 코드 `1dcb10ba`. 새 작업 경로는 `/home/chokwans99/PSD-local-followup-20261017`, branch는 `exp/duet-local-followup-20261017`다. Runtime3파일과 portable 회귀 검사1파일을 변경·추가했다. Production 브랜치/이전 실험 브랜치는 건드리지 않았다. 새 옵션·알고리즘/파라미터 변경·전송 형식 변경·KV 의미 변경은 없다.

`PREP_RUNTIME.patch`만 채택 대상이다. `BUCKETS_RUNTIME.patch`, `OVERLAP_RUNTIME.patch`는 미채택 증거로 남겼다. 서로 독립한 후보이며 누적 성능으로 해석하지 않는다. 새 변경을 이전1.43%와 단순 합산하지 않는다.

재현 입력/실행 plan·run_local.py·queue_local.py·source hash·검사 로그와 원시 archive를 함께 보존했다. Root 경로·CPU pin·GPU쌍·port가 local orion 전용이므로 다른 서버에 그대로 launcher를 실행하지 않는다. Code snapshot `prep`와 현재 runtime94개 파일이 byte 단위로 같은지 검증했다.

1024토큰/전체 SpecBench 검증과 local 파라미터 재탐색은 이번 작업 범위에 넣지 않았다. GPU4–7의 실험은 종료하고 반환했다. Commit/push하지 않았다.
