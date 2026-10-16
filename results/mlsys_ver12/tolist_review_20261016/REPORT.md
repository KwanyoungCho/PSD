# `tolist()` 제거 시험 — 무효 / 미채택

**사용자 지시에 따라 성능 개선 근거가 없는 이번 후보는 무효 처리한다. 원래 `MLSys-ver1.2` runtime으로 후속 실험을 계속한다.**

현재 branch의 `ssd/ssd`는 시험 전과 모든 파일 해시가 동일하다. 수정은 detached 시험용 복사본 한 파일에만 있었다. Patch·검사·원시 결과는 미채택 근거로 보존하며 production 적용 대상으로 사용하지 않는다.

[설계와 실제 수정](README.md) · [미채택 patch](CANDIDATE.patch) · [전체 수치](SUMMARY.json) · [완료 검사](CHECKS.json)

로컬 LayerSkip-Llama2-13B + llama-68m, targetGPU4·5/draft6. B1, BF16, T0.7, exit26, K1=K2=3, DS위치당 root3/PSroot9, W9/9, C3, G=M40, last-depth 경쟁. 같은12입력×2회로 변경 전→후→후→전 네 process를 실행했다.

두 A/B 비교의 공통 유효 step은3,178개다. Warmup·초기10step·terminal/cap·clip·실제 compile/capture 영향만 제외한다. 느린 step을 임의 제외하지 않는다.12 UID로 묶은 paired bootstrap은 passes와 두 process 비교를 함께 유지한다.

## 직접 latency 비교

| 작업 | 변경 전 (ms) | 변경 후 (ms) | 차이 (ms) | 차이의95% 구간 (ms) |
|---|---:|---:|---:|---|
| 위치 map 갱신 함수 CPU 반환 | 0.153138 | 0.136262 | -0.016876 | -0.024105 ~ -0.009416 |
| 뒤의 P2 상태 확인 CPU 반환 | 0.248052 | 0.261466 | +0.013414 | -0.001670 ~ +0.028893 |
| PS 후보·layout·args 구성 | 1.326570 | 1.334913 | +0.008343 | -0.062039 ~ +0.079938 |
| Proxy 수신→PS graph 시작 준비 합 | 3.152530 | 3.193192 | +0.040663 | -0.135141 ~ +0.218443 |
| PS 세 forward+sampling graph | 1.604607 | 1.606490 | +0.001883 | -0.000431 ~ +0.004238 |
| Proxy 수신→cache merge PS 전체 | 4.799376 | 4.843121 | +0.043745 | -0.134456 ~ +0.223839 |
| Target step 전체 | 29.463805 | 29.512957 | +0.049151 | -0.201742 ~ +0.264509 |

위치 map 갱신 함수는 평균0.016876ms 줄었지만 뒤 상태 확인은0.013414ms 늘었다. Readback을 생략하면 미완료 GPU 작업이 뒤의 scalar/state readback에서 대기할 수 있다. 이 패턴과 일치하는 관측이지만 모든 차이를 고립된 동기화 이동의 causal 효과로 확정하지 않는다.

PS 준비 전체는+0.040663ms, PS 전체는+0.043745ms다. 두 반복 모두 PS 전체 평균이 감소하지 않았고95% 구간은0을 포함한다. 이 측정에서는 확실한 성능 개선을 입증하지 못했으므로 미채택한다. 약0.9%의 평균 증가 자체도 통계적으로 확정된 회귀라고 주장하지 않는다.

| 현재 source | 공통 step | PS 준비 전→후 (ms) | PS 전체 전→후 (ms) |
|---|---:|---|---|
| P1 hit | 1482 | 3.178 → 3.168 | 4.824 → 4.818 |
| P2 hit | 1184 | 3.129 → 3.233 | 4.779 → 4.886 |
| Miss | 512 | 3.133 → 3.174 | 4.775 → 4.818 |

| 반복 | 위치 map CPU 차이 (ms) | PS 준비 차이 (ms) | PS 전체 차이 (ms) |
|---|---:|---:|---:|
| 1: before1 / after1 | -0.016177 | +0.041780 | +0.048505 |
| 2: before2 / after2 | -0.017574 | +0.039545 | +0.038984 |

## 제거 사실과 정확성

고정 count tensor의 같은-process 측정에서는 위치 목록 DtoH가5회→0회였다. 동작을 정확히 생략했음은 확인된다. 해당 작업의 batch-fenced 시간은 0.072003→0.064499ms다. 이 작은 측정은 순서대로 실행한 보조 측정이며 전체 엔진의 DtoH 제거나 PS latency 개선 근거로 사용하지 않는다.

Candidate contract4개+기존 P2 준비12개+dispatch9개 검사25개 통과. 같은 GPU 위치/rope/mask와 실제 root/padding 경계를 확인했고, eager fallback은 기존 host 목록/metadata를 복원했다. CPU audit에서 매 step 후보 목록의 host materialization 여부를 확인했다.

출력48/48 동일. 같은 모든 measured step의 source/AL/검증 폭/cache hit/실제 반환량/cap·clip도 동일했다. Raw summary AL2.305722와 hit82.397820%는 네 실행에서 동일하며 새 알고리즘 개선 수치가 아니다.

초기 test 실행은 SSD_HF_CACHE 미설정으로 import 실패했고, env를 설정한 후 검사를 통과했다. 최초 환경 오류 로그도 보존했다. Model process는 네 번 모두 정상 종료했으며 assigned GPU의 foreign process는 telemetry에서 관측되지 않았다.

## 후속 실험 기준

`MLSys-ver1.2`와 현재 작업 디렉터리를 그대로 사용한다. 미채택 수정으로 default·수식·파라미터·tree 정책을 바꾸지 않는다. 이번 시험으로 과거9ms 준비 spike의 직접 원인을 확정하지 않는다. GPU polling은 CPU scheduling/API 대기를 측정하지 않았고, 현재 짧은 반복은 다른 모델/B>1의 성능 확정이 아니다.

시험용 before/after worktree는 사용자 무효 처리 지시에 따라 정리한다. 원시 run과 미채택 patch/검사는 기록으로 남긴다. 다른 사용자의 프로세스는 종료하지 않았다.
