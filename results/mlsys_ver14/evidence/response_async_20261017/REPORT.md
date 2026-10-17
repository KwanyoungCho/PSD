# Target 응답 우선 처리 추가 최적화 — ver1.3 기준 임시 브랜치

상태: **선언한32개 모델 실행과 분석 완료. Host-list 13B의 출력 불일치는 보존하고 해당 성능 비교를 제외했다. 기존 `MLSys-ver1.3@5642de04`는 변경하지 않았다.**

## 질문에 대한 답과 변경 범위

Cache hit 응답을 먼저 보내면 target 대기를 줄일 기회가 있다. 1.3은 이미 이전 draft KV 복원을 응답 뒤로 옮겼다. 이번에는 남은 응답 준비와 전송 완료 대기를 각각 분리했다. 확률·root/위치/tree 정책·K·W·M은 그대로다. **임시 브랜치 HEAD는 최종적으로 두 모델 공통 개선을 확인한 `prepack` runtime으로 정리했다.** 미채택 async/early/host-list 후보는 독립 patch와 원시 기록으로 보존하며 production에는 반영하지 않았다.

| 후보 | 기존 → 시험 동작 | 정확성 조건 |
|---|---|---|
| 전송 비동기화 | `dist.send`의 즉시 Work.wait를 없애고 두 payload를 isend | Work와 tensor를 보유하고 다음 요청에서 buffer 재사용 전 wait. 전송 중 tensor를 덮어쓰지 않음 |
| 토큰 우선 | 작은 token/topology packet enqueue 뒤 parent-q gather, 응답 enqueue 뒤 glue 입력 준비 | 동일 header/q shape·순서, 동일 proposal·RNG. Miss JIT 전 KV 복원 유지 |
| 사전 포장 | DS/PS build 중 모든 root의 작은 topology snapshot, hit에서는 해당 row 선택·검사 | G=M에서만 사용. Full-vocabulary q는 hit root만 gather. 추가 DS/PS 비용도 측정 |
| CPU 정수 목록 | 이미 CPU로 복사한 packet을 한 번에 list로 변환한 뒤 기존 검사 | GPU→CPU 복사 횟수는 동일. CPU tensor 인덱싱·개별 정수 변환 비용 감소. 한 줄의 runtime 변경 |

설치된 PyTorch2.8의 `send()`가 `isend()` 후 `work.wait()`를 호출함을 소스에서 확인했다. 비동기화는 전송량을 줄이지 않으며, 현재 target은 토큰/topology와 q를 받은 후 다음 forward를 시작한다. Draft의 다음 작업과 전송을 겹치더라도 target의 응답 전 준비 비용이 사라지는 것은 아니다.

## 전체 step 성능 — Profile OFF

같은 출력 토큰·source/cache-hit/valid-k/수락 길이의 step을 대조한다. Cap/clip·부분 batch 및 동일 request/batch/source 구성 median의2배 초과 제외 후 양 arm 교집합이다. Raw 비교도 함께 저장했다.

| Target +68m | 후보 | 전체 step 전→후(ms) | 감소율 [95% CI] | 반복 a / b | 독립 wall 감소 | 판단 |
|---|---|---:|---:|---:|---:|---|
| 7b | 전송 비동기화 | 30.996→31.000 | -0.01% [-0.32, +0.30] | -0.24% / +0.22% | -0.01% | 일관된 개선 미확정 |
| 7b | 토큰 우선 전송+비동기 | 30.996→31.031 | -0.11% [-0.27, +0.03] | -0.02% / -0.20% | -0.12% | 일관된 개선 미확정 |
| 7b | Tree 정보 사전 포장 | 30.996→30.553 | +1.43% [+1.25, +1.60] | +1.26% / +1.60% | +1.43% | 이번 조건에서 개선 |
| 7b | CPU 정수 목록 파싱 | 31.113→30.972 | +0.45% [+0.00, +0.88] | +0.91% / -0.00% | +0.45% | 일관된 개선 미확정 |
| 13b | 전송 비동기화 | 44.083→43.999 | +0.19% [-0.12, +0.51] | -0.21% / +0.59% | +0.19% | 일관된 개선 미확정 |
| 13b | 토큰 우선 전송+비동기 | 44.083→43.943 | +0.32% [+0.05, +0.63] | +0.08% / +0.56% | +0.32% | 이번 조건에서 개선 |
| 13b | Tree 정보 사전 포장 | 44.083→43.637 | +1.01% [+0.82, +1.22] | +0.22% / +1.80% | +1.01% | 이번 조건에서 개선 |
| 13b | CPU 정수 목록 파싱 | N/A | N/A | 원시 불일치 보존 | N/A | 출력·수락 기록 불일치로 전체 arm 제외 |

이번 조건의 개선 판정은 CI 하한>0, 두 process 반복 모두 감소, 독립 wall 감소를 함께 요구한다. CI는 고정12 UID cluster bootstrap이다. Process 변동 전체를 대표하지 않으므로 반복별 부호도 확인한다. 전체 `LLMEngine.step()`의 별도 perf_counter는 GPU 동기화 없이 계측했고, 모든 warm/measured step의 순서·native 시간·source·AL이 원시 기록과 맞는지 검사했다.

[전체 step 비교 그림](LATENCY_COMPARISON.pdf), [JSON](STEP_COMPARISONS.json), [독립 wall 비교](WALL_COMPARISONS.json), [P1/P2/miss CSV](SOURCE_LATENCIES.csv).

변경한 draft 응답 준비와 대응하는 노출 비용을 별도로 확인한다. 아래 verification은 target 호출·수락 구간이고 그 밖은 요청·응답·스케줄링 등이다. Transformer의 계산량은 같다.

| Target | 후보 | Target 호출·수락 전→후(ms) | 그 밖의 step 전→후(ms) |
|---|---|---:|---:|
| 7b | Tree 정보 사전 포장 | 27.429→27.464 | 3.567→3.089 |
| 7b | CPU 정수 목록 파싱 | 27.498→27.531 | 3.615→3.442 |
| 13b | Tree 정보 사전 포장 | 40.251→40.243 | 3.832→3.393 |

## 응답 대기와 forward 사이 공백 — 별도 Profile ON 진단

요청 send 종료→q까지 수신 완료를 response wait, 이전 final logits 종료→다음 graph_pre 시작을 전체 공백으로 측정한다. 첫10step·cap/clip 제외. 이 계측값을 비계측 절감량에 더하지 않는다.

| Target | 후보 | 응답 대기(ms) | 전체 forward 공백(ms) | 응답 완료→forward(ms) |
|---|---|---:|---:|---:|
| 7b | ver1.3 | 2.700 | 6.231 | 1.277 |
| 7b | 전송 비동기화 | 2.749 | 6.275 | 1.284 |
| 7b | 토큰 우선 전송+비동기 | 2.618 | 6.137 | 1.224 |
| 7b | Tree 정보 사전 포장 | 2.425 | 6.196 | 1.367 |
| 7b | CPU 정수 목록 파싱 | 2.379 | 5.785 | 1.238 |
| 13b | ver1.3 | 2.732 | 7.124 | 1.809 |
| 13b | 전송 비동기화 | 2.856 | 7.385 | 1.855 |
| 13b | 토큰 우선 전송+비동기 | 2.588 | 7.007 | 1.755 |
| 13b | Tree 정보 사전 포장 | 2.274 | 6.682 | 1.847 |
| 13b | CPU 정수 목록 파싱 | 2.361 | 6.851 | 1.922 |

[Profile 구간 그림](PROFILE_BREAKDOWN.pdf), [전체 source별 세부 구간](PROFILES.json). 부모/자식 marker는 중첩되어 합산하지 않는다. Async 후보의 send marker는 enqueue 시간이며 통신 완료 시간으로 해석하지 않는다.

7B의 profile-ON 전체 공백은6.231→6.197ms로 거의 중립이었다. 응답 대기는 줄었으나 그 밖의 profiled 구간 평균이 증가했다. 이를 숨기거나 profile에서0.44ms가 줄었다고 설명하지 않는다. 효과 크기의 주 근거는 별도 profile-OFF 전체 step과 독립 wall이고, profile은 바뀐 draft 포장 비용과 응답 대기의 방향을 확인하는 진단 자료다. 단일 profile 간 차이의 원인을 모두 분리한 것은 아니다.

기준 hit 준비는 약1.75–1.87ms, 그 안의 topology 포장·검사는 약0.56–0.71ms였다. q 수신은 약0.15–0.22ms라서 전송 비동기화만의 여지는 제한적이었다. 사전 포장은 이 일을 DS/PS로 옮기므로 응답 감소와 전체 step 개선을 따로 판단해야 한다.

CPU-only 메커니즘 검사에서39노드 parse/validate는 tensor 원소 접근0.450ms→정수 목록0.047ms였다. 이는 GPU/NCCL이 없는 원인 분리 실험이며 full-model 개선율로 인용하지 않는다. [MICRO_HOST_PARSE.json](MICRO_HOST_PARSE.json).

사전 포장은 DS/PS에서 다음 비용을 추가한다. 해당 marker가 존재한 호출의 평균이며 source별 상세 표본 수는 PROFILES.json의 component_samples에 있다.

| Target | DS topology 준비(ms) | PS topology 준비(ms) |
|---|---:|---:|
| 7b | 0.233 | 0.110 |
| 13b | 0.215 | 0.100 |

Root당 metadata는 `8×(3+4M_wire)` bytes다. M_wire40에서는1.304KB/root이며, DS bucket 상한123 root 슬롯을 준비하면 약160.4KB, PS9 roots는11.7KB다. Full-vocabulary logits는 사전 포장하지 않는다. Target↔draft 전송량도 기존과 같다. 추가되는 것은 build 시점의 로컬 GPU→CPU metadata 복사와 CPU 객체 준비다. 큰 R/M이나 DS/PS 여유가 부족한 설정에는 비용이 달라지므로 이번 개선율을 그대로 일반화하지 않는다.

## 검증·조건·미반영 범위

- 응답 우선+async runtime은 GPU/CPU 회귀68개, 사전 포장 runtime은69개 통과, 실패·오류·skip0. 두 suite는 공통 검사가 겹친다. Host-list의 새 CPU 검사2개도 통과했고 모든 해당 모델 실행의 출력·수락 parity를 별도로 확인했다.
- 기준과 일치한 candidate 실행들의 출력 456건, 수락 event 35439개. 실패한 비교를 여기에 합치지 않았고 반복을 독립 문항 수로 세지 않는다. 각 PARITY.json과 raw를 보존했다.
- 전체32실행: profile10 + profile-OFF22. 첫 main 캠페인은 before/send/early/prepack/prepack/early/send/before. Host-list는 main의 마지막 before를 첫 A로 재사용하여 A→B→B→새 A를 구성한 별도 ABBA다. 같은 baseline을 중복된 새 실행으로 세지 않았다.
- Host-list 13B의 첫24입력 실행은 두 번째 pass의 UID501_t0, 출력 token index4(recovery/bonus)에서 처음 달랐다. 다음 step3은 기준 P2 hit→후보 miss로 바뀌었다. 마지막3개 계획은 재현성 진단으로 전환해 원래 실패를 대체하지 않고 모두 보존했다. 상세 [PARITY_FAILURE.json](PARITY_FAILURE.json), [REPEATABILITY.json](REPEATABILITY.json). CPU 정수 값의 동등성은 검사했지만 첫 recovery-token 차이의 원인은 확정하지 않았다. 이를 파서의 수학적 오류 또는 무해한 수치 잡음으로 단정하지 않으며 이 후보는13B 채택 대상에서 제외한다.
- 이 불일치 관측만으로 prepack이 기존 엔진의 재현성 원인을 해결했다고 주장하지 않는다. 선택한 후보의 정확성 주장은 이번 출력·수락 parity와 검사 통과 범위다.
- eslab18/nova RTX3090, 7B target2/draft3, 13B target4·5/draft6. 13B의 target2장은 tensor parallel이며 batch는1이다. GPU0·1 및 외부 프로세스는 종료하거나 사용하지 않았다.
- 두 target은 full BF16 weights, draft JackFram/llama-68m. B1, T0.7, seed9101, K1=K2=3, C3, G=M40. DS 위치당3 roots, PS9 roots, DS 후속 입력 상한9/PS 입력9. DS 첫 round는 root 수에 따라 달라진다.
- Exit21/26(0-based), 최대 문맥2048, 입력 cap512, 출력 cap192, warm3. 고정 Spec-Bench12턴(6과제)×2회/process이며 full480문항 AL/TPS 평가·파라미터 최적화가 아니다.
- 제외된 긴 step 개수는 [EXCLUSIONS.json](EXCLUSIONS.json)에 run별로 기록했다. Profile-OFF22실행의 긴 step 제외 합계는 0개다. 원시 시간·출력은 보존했고 준비 시간을 retained step에서 임의 차감하지 않았다.
- 회귀 driver의 첫 시도는 반복 interpreter 초기화 비용을 줄이기 위해 중단하고 partial 로그를 보존했다. 일반 검사 묶음과 import-time fixture flags를 쓰는 묶음을 분리해 최종 검사를 다시 완료했다. 모델 실행 중단이나 실패 숨김이 아니다.
- B>1, greedy/T=0 full-model, 70B/Blackwell의 새 성능 검증은 하지 않았다. Target 전반 forward와 q 수신을 별도 stream으로 겹치는 consumer 변경은 이번 실험에 넣지 않았다.

## 다른 작업자 검토·재현

- 기준: `MLSys-ver1.3@5642de04`. 메인 임시 브랜치 `exp/duet-response-async-20261017`. 사전 포장 commit `7f77ea7f`, 한 줄 host-list commit `e57762ae`는 별도 실험 가지다. [LOCAL_VARIANTS.json](LOCAL_VARIANTS.json)에 local commit과18번 source-only branch/hash 대응이 있다.
- 각 독립 후보는 `ASYNC_SEND.patch`, `EARLY_PAYLOAD.patch`, `PREPACK_TOPOLOGY.patch`, `HOST_INTEGER_PARSE.patch`다. 모두1.3 기준이며 서로 누적 적용하지 않는다. 검토 전 `git apply --check`와 해당 정확성 검사를 실행한다.
- 기존1.3은 보존했다. 개선 후보도 임시 브랜치에만 두며 이번 요청으로 production에 자동 반영하지 않는다.
- 새 Git checkout의 이 보고서 폴더(`raw/`가 없는 상태)에서 `tar -xJf RAW_RESULTS.tar.xz` 후 `python analyze.py --raw raw`, `python plot_results.py`, `python finalize.py`로 재분석한다. Numpy/matplotlib/pypdf 환경만 필요하고 원본 서버·모델·GPU는 필요 없다.
- 결과·hash manifest·검사 로그·telemetry를 archive에 보존한다. 원격 재실행 controller는 nova 절대 경로를 포함하므로 다른 서버에서 그대로 실행하지 말고 경로·GPU·port를 검토한다.
- 실행 코드만 이식할 때는1.3 위에 독립 `7f77ea7f` 또는 `PREPACK_TOPOLOGY.patch`를 검토한다. `b86d4abc`는 async prototype을 정리한 커밋이라1.3에 단독 cherry-pick하지 않는다. 실험 가지들을 history 보존용 ours merge로 연결했고, HEAD의 runtime은 선택한 prepack과 byte 단위로 같다.
- [FINAL_CHECKS.json](FINAL_CHECKS.json)은32실행의 source/bench 동일성,69개 검사와 선택 runtime의 동일성, 독립 wall45,451 step, GPU telemetry, 원래1.2/1.3 보존을 검사했다. `python audit_results.py --output /tmp/response-review-checks.json`으로 자료를 다시 확인할 수 있다. `--check-runtime`은 선택한 코드 checkout에서만, `--check-protected`는 원래 두 worktree가 있는 이 서버에서만 사용한다.
- [ARCHIVE_CHECK.json](ARCHIVE_CHECK.json)에 원시649파일의 개별 SHA256과 archive SHA256을 기록했다. [ARTIFACTS.json](ARTIFACTS.json)은 공개 보고서·그림·스크립트·patch의 최종 hash다.
